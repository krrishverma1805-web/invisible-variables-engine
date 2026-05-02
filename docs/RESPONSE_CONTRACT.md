# Response Contract — Invisible Variables Engine

**Version:** v1
**Effective date:** 2026-04-28
**Owner:** ML lead (per plan §122)
**Audience:** integrators, auditors, and on-call engineers

This document is the **authoritative spec** for what IVE guarantees in its
API responses, what it explicitly does not guarantee, and how each
guarantee is enforced in code. When the plan says "documented in
`docs/RESPONSE_CONTRACT.md`," this is that document.

If you are about to trust an IVE number — for a clinical decision, an
operational change, a regulatory filing, a public dashboard — read the
relevant section first.

---

## Table of contents

1. [Reproducibility](#1-reproducibility)
2. [Confidence interval methodology](#2-confidence-interval-methodology)
3. [FPR (false-positive-rate) guarantees](#3-fpr-guarantees)
4. [`explanation_source` semantics](#4-explanation_source-semantics)
5. [Data retention](#5-data-retention)
6. [API versioning + deprecation](#6-api-versioning)
7. [Structured-logging key catalog](#7-structured-logging-keys)
8. [Known incompatibilities](#8-known-incompatibilities)
9. [Model migration runbook](#9-model-migration-runbook)
10. [Deploy ordering](#10-deploy-ordering)
11. [Auth & rate limiting](#11-auth-rate-limiting)
12. [API stability commitments](#12-api-stability-commitments)
13. [Data egress inventory](#13-data-egress-inventory)
14. [Sensitivity model](#14-sensitivity-model)
15. [Self-hosted-LLM compatibility](#15-self-hosted-llm-compatibility)
16. [Selective-inference methodology](#16-selective-inference-methodology)
17. [Operational posture](#17-operational-posture)
18. [Cost model](#18-cost-model-groq-pricing-as-of-2026-04-27)
19. [Data-lineage tracking](#19-data-lineage-tracking)
20. [Multi-tenancy](#20-multi-tenancy)
21. [Release-train cadence](#21-release-train-cadence)
22. [Multiclass classification support](#22-multiclass-classification-support)
23. [`holdout_uplift` versioning](#23-holdout_uplift-versioning)
24. [CI gates](#24-ci-gates)

---

## 1. Reproducibility

**Phase A guarantee.** Given the same `(random_seed, study_seed, cv_seed)`
tuple, the same dataset bytes, and the same code revision, every IVE
output column is byte-identical across runs.

| Axis | Setting | Default | Notes |
|---|---|---|---|
| Random seed | `MLSettings.random_seed` | `42` | Drives data splits, bootstrap resamples, sklearn estimators. |
| Study seed | `MLSettings.hpo_study_seed` (Phase B) | `random_seed` | Hyperparameter-search axis (Optuna TPESampler). Set explicitly when reproducibility across HPO matters. |
| CV seed | `MLSettings.cv_seed` (Phase B) | `random_seed` | K-fold cross-fitting splits. Defaults to `random_seed`; override only when comparing two analyses with intentionally different fold assignments. |

**Phase A scope:** the `random_seed` axis fully determines outputs because
HPO and cross-fitting are not yet shipped. When Phase B lands, the contract
expands to the three-axis tuple above.

**Phase A LLM determinism.** With `GROQ_TEMPERATURE=0.0` (default), the
same prompt template + same model identifier produces the same output
modulo Groq's internal kv-cache nondeterminism (an upstream artifact
outside our control). Our golden snapshots capture the *prompt* string,
not the model's *response*; the response is checked by `composite_validate`
against the input facts, not by string equality.

**Reproducibility CI.** `tests/statistical/test_reproducibility.py` runs
the same experiment twice with fixed seeds and asserts byte-identical
outputs across all DB columns. This is the gate on the contract; if it
fails, the seed contract is broken.

---

## 2. Confidence interval methodology

**Phase A:** confidence intervals are **not** populated. The
`latent_variables.confidence_interval_lower/upper` columns exist in the
schema but stay `NULL` until Phase B4 ships.

**Phase B4 (planned):**
- Method: BCa (bias-corrected and accelerated) bootstrap when subgroup
  N ≥ 100; percentile bootstrap as a fallback below that threshold (BCa
  is unstable on small samples).
- Coverage target: ~95% empirical coverage, validated by
  `tests/statistical/test_effect_size_ci_coverage.py`.
- Selection-bias correction: K=5 cross-fitting (per plan §96). The `≥3 of K`
  stability filter is itself a discovery-conditioning step, so reported CIs
  remain mildly anti-conservative (~2–4% width contraction). Documented
  here, not silently absorbed.
- Off-grid configurations (cross-fit on/off, different bootstrap iteration
  counts) trigger an `effect_size_ci_methodology_off_grid` warning event;
  CIs in that case are computed but flagged.

---

## 3. FPR guarantees

**Phase A:** FPR control uses fixed bootstrap-presence thresholds (0.7
production / 0.5 demo) — no empirical calibration. Treat these numbers as
heuristics, not guarantees.

**Phase B6 (planned):** thresholds are loaded from
`data/calibration/stability_thresholds.json`, calibrated offline per
`(n_rows, mode, problem_type, bootstrap_iterations, min_subgroup_size, min_effect_size)`
tuple to yield FPR ≤ 5% on white-noise datasets.

**Calibration fallback chain** (per plan §149):

| Trigger | Strategy | Behavior |
|---|---|---|
| Active config matches grid | `"table"` | Use exact threshold from JSON. |
| Active config off-grid + `_on_mismatch="adaptive"` | `"adaptive"` | Run online permutation calibration (n=200 per plan §97). 5–10 min runtime. |
| Active config off-grid + `_on_mismatch="fixed"` | `"fixed"` | Use `DetectionSettings.default_stability_threshold`. Last-resort emergency override. |

**Honest scope.** The offline calibration is computed on synthetic
Gaussian noise. Real-world feature correlations, target skew, and outliers
shift the FPR curve; the offline grid is approximate. The `"adaptive"`
mode tightens the guarantee at the cost of per-experiment runtime. The
`"fixed"` mode bypasses both — use only when calibration data is
unavailable or known-bad.

**Cross-fit interaction** (plan §188): cross-fit residuals have a slightly
different distribution than single-model residuals, so the table must be
regenerated when cross-fitting lands. The committed JSON's
`config_grid.cross_fit_mode` field declares which mode it was calibrated
for; mismatch triggers `"adaptive"` fallback.

---

## 4. `explanation_source` semantics

Every endpoint that surfaces an explanation also returns
`explanation_source`, `llm_explanation_pending`, and
`llm_explanation_status`. The combinations have these meanings:

| `llm_explanation_status` | LLM text present | `explanation_text` content | `explanation_source` | `llm_explanation_pending` |
|---|---|---|---|---|
| `ready` | yes | LLM-polished prose | `llm` | `false` |
| `ready` | no | Rule-based prose (defensive fallback) | `rule_based` | `false` |
| `pending` | — | Rule-based prose (LLM still queued) | `rule_based` | **`true`** |
| `failed` | — | Rule-based prose | `rule_based` | `false` |
| `disabled` | — | Rule-based prose | `rule_based` | `false` |

**UI polling contract.** When `llm_explanation_pending=true`, the UI
should poll the same endpoint every 10 s for up to 5 minutes. After that
window, treat the explanation as terminal (likely `failed` or stuck
`pending`) and stop polling.

**`failed` vs `disabled` distinction.**
- `failed` — the LLM was attempted but the output couldn't be validated
  (numeric grounding, banned phrases, transient API outage that exhausted
  retries). The audit log records the failure reason.
- `disabled` — the LLM was deliberately not attempted. Reasons include:
  global flag off (`LLM_EXPLANATIONS_ENABLED=false`), per-column
  protection (`pii_protection_per_column`), or token budget exceeded
  (`token_cap_predictive`).

**Endpoints that emit these fields:**
- `GET /api/v1/experiments/{id}/summary`
- `GET /api/v1/experiments/{id}/latent-variables`
- `GET /api/v1/latent-variables/`
- `GET /api/v1/latent-variables/{id}`

---

## 5. Data retention

| Storage | TTL | Backup posture | Notes |
|---|---|---|---|
| Postgres user data (annotations, feedback, audit logs) | Indefinite | Managed snapshots + 7-day PITR (RPO 5 min, RTO 30 min in production); `pg_dump` + WAL archiving for self-hosted | See `docs/runbooks/postgres_backup.md` (planned). |
| Redis db 0 (Celery broker) | Per-task | Best-effort persistence (managed snapshot) | Loss → operator marks orphan experiments `failed` via admin endpoint. |
| Redis db 1 (Celery results) | `celery_result_expires` (default 24 h) | Same as db 0 | Regenerable. |
| Redis db 2 (LLM cache) | `LLM_CACHE_TTL_SECONDS` (default 7 days) | None — fully regenerable | On dataset/experiment delete, the cascade purges keys via the per-entity index set (per plan §34). |
| Calibration JSON (`data/calibration/*.json`) | Tied to git history | Versioned in repo | Operator-triggered regen via `make calibrate-stability`; CI does not run it. |
| LLM-generated text (`latent_variables.llm_explanation`, etc.) | Indefinite | Same as Postgres backups | Cleared on row delete via FK cascade. |

**Cascade-delete invariant.** Deleting a dataset row deletes:
1. Every experiment FK'd to it (cascade).
2. Every LV / pattern / event / trained_model FK'd to those experiments
   (cascade).
3. Every `dataset_column_metadata` row (cascade — added in PR-3).
4. Every Redis-cached LLM response indexed under
   `("experiment", str(experiment_id))` (per `RedisLLMCache.delete_for_entity`).

The cascade is enforced at the DB layer (FKs with `ON DELETE CASCADE`) so
even raw `DELETE FROM datasets WHERE id = ...` from a DBA shell preserves
consistency.

---

## 6. API versioning

**v1 stability commitment.** From the v1 GA date (2026-05-15) for
**12 months**, every change to v1 is additive only. Concretely:

| Change | Allowed in v1? |
|---|---|
| Add a new optional response field | Yes |
| Add a new endpoint | Yes |
| Add a new optional query parameter | Yes |
| Add a new error code for previously-unhandled cases | Yes |
| Performance / latency improvements | Yes |
| Remove a response field | **No** — bump to v2. |
| Narrow a field's type (e.g. `str` → `enum`) | **No** — bump to v2. |
| Change semantics of an existing field | **No** — version the field name. See `holdout_uplift` in §8. |
| Tighten an error code (e.g. 422 → 400) | **No** — bump to v2. |
| Add a new required request field | **No** — bump to v2. |
| Increase auth requirements on an existing route | **No** — bump to v2. |

**v2 introduction path.** v2 launches as a parallel router (`/api/v2/`).
v1 sunset is announced ≥6 months in advance via a `Deprecation: true`
header on every v1 response. Active v1 keys receive an email notice on
sunset announcement and again 30 days before sunset.

---

## 7. Structured-logging keys

Every log line emitted by IVE code uses `structlog` with a stable key
catalog. Log consumers (Datadog, CloudWatch, Loki) can rely on these keys
without parsing free-form messages.

| Key | Type | Source | Notes |
|---|---|---|---|
| `dataset_id` | UUID string | All dataset-scoped operations | Always present when a dataset is in scope. |
| `experiment_id` | UUID string | All experiment-scoped operations | |
| `lv_id` | UUID string | LV-scoped operations | |
| `pattern_id` | UUID string | Pattern-scoped operations | |
| `request_id` | string | API requests + chained Celery tasks | Propagated via Celery `task headers` per §62. |
| `task_id` | Celery task UUID | Celery-task scope | |
| `trace_id` | OTel trace ID | Phase C | |
| `phase` | int (1–5) | Pipeline-phase events | |
| `subphase` | string | Sub-step events | e.g. `hpo`, `ensemble_fit`. |
| `event_type` | string | All structured events | One of `started` / `completed` / `failed` / `skipped`. |
| `latency_ms` | int | I/O ops, Groq calls | |
| `tokens_in` / `tokens_out` | int | LLM calls | |
| `model` | string | LLM calls | The Groq model identifier at call time. |
| `cache_status` | string | Cache lookups | `hit` / `miss` / `bypass`. |
| `validation_failure_reason` | string | LLM validator failures | Populated when `composite_validate` rejects an output. |
| `circuit_breaker_state` | string | Breaker transitions | `closed` / `open`. |

**Redaction policy** (planned, plan §112): when the active dataset has
any column marked `non_public`, log lines emitted from the
`run_llm_enrichment_async` path strip raw values and feature names
through a structlog processor before the log record leaves the process.

---

## 8. Known incompatibilities

### 8.1 `holdout_uplift` semantic shift (Phase B)

Pre-B3, `experiments.holdout_uplift` is computed against a linear
baseline. Post-B3 (ensemble landing), the canonical metric is computed
against the ensemble baseline.

**Mitigation:** the old field stays populated by the *old computation*
indefinitely (no aliasing — old clients see the numbers they always saw).
A new field `final_holdout_uplift` carries the post-B3 ensemble-baseline
value. Responses include a `Deprecation: true` header pointing readers to
the new field. v1 will sunset `holdout_uplift` ≥6 months after B3 GA.

### 8.2 Multiclass classification

Multiclass targets route to `XGBClassifier` for prediction but **skip
residual-based detection**. Supported features:

- Trained models per fold
- Phase 5 holdout uplift (log-loss)
- HPO over the multiclass search space (Phase B)
- Ensemble stacking with a multiclass meta-learner (Phase B)
- LV apply for non-residual-derived LVs (temporal patterns, etc.)

Skipped: subgroup discovery, HDBSCAN clustering on residuals, SHAP
interaction mining, variance regimes, B4 effect-size CIs.

`ExperimentResponse.multiclass_supported_features: list[str]` makes this
machine-readable.

---

## 9. Model migration runbook

When Groq deprecates `llama-3.3-70b-versatile` (typical cadence: quarterly
to semi-annual), follow this procedure to migrate to the replacement
without surprising downstream consumers:

1. **Pin the current state.** Ensure `model_capabilities.yaml` records
   the current `pinned_at` date and the current `LLM_PROMPT_VERSION`.
2. **Run the behavioral suite against the candidate replacement.** The
   suite (`tests/llm/behavioral_corpus.json`, planned) sends 50 canonical
   prompts and asserts numeric grounding pass rate ≥99%, validation pass
   rate ≥95%, banned-phrase rate ≤1%.
3. **A/B against production cache misses for 1 week.** Run both models in
   shadow on the same payloads; compare validator pass rates and reader
   feedback (the `explanation_feedback` table).
4. **Pin the replacement.** Update `GROQ_MODEL` in production env config.
5. **Bump `LLM_PROMPT_VERSION`** to invalidate the cache. Schedule the
   bump for low-traffic hours; the first run after the bump is full-cost.
6. **Update `model_capabilities.yaml`** with the new `pinned_at` date.

For self-hosted swaps (vLLM, etc.), additionally run validator-profile
qualification (per plan §141) before promoting; different model families
have different failure modes, so banned-phrase / hedge-marker / numeric-
tolerance tuning is per-model.

---

## 10. Deploy ordering

Every alembic migration in `alembic/versions/` carries a `deploy:`
directive in its docstring on its own line, matching:

```
^deploy: (schema-first|code-first|same-release)( \| reason: .+)?$
```

| Directive | Meaning |
|---|---|
| `schema-first` | Migration applies before code deploy. Old code reads the new column at startup. |
| `code-first` | Code deploys first; migration applies later. Column stays NULL until backfilled. |
| `same-release` | Both must land in the same atomic deploy. |

**CI gate** (planned, plan §148): `.github/workflows/migrations_check.yml`
parses each migration file's docstring, extracts the directive, fails the
PR if the directive is missing or unrecognized.

**Single-head invariant.** Production startup runs `alembic upgrade head`;
two heads is a deploy-blocker. CI gates on `alembic heads | wc -l == 1`.

**Phase A migrations** (in apply order):
1. `add_lv_llm_columns` — code-first
2. `add_experiment_llm_columns` — code-first
3. `add_explanation_feedback_table` — code-first
4. `add_dataset_column_metadata` — schema-first

---

## 11. Auth & rate limiting

**Authentication.** API-key based. Header: `X-API-Key`. Keys hashed at
rest in the `api_keys` table. Per-key scope model (`read | write | admin`)
and audit log added in PR-2.

**Per-key rate limiting.**

| Endpoint family | Default ceiling | Setting |
|---|---|---|
| Catch-all API rate | 100 req/min per key | `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW` |
| Feedback submissions | 200 req/hour per key (planned) | `RATE_LIMIT_FEEDBACK_PER_HOUR` |

**LLM concurrency is global, not per-key** (plan §181). A paid API key
does **not** get faster LLM enrichment — Groq tier limits are global to
the deployment. To accelerate enrichment under load, raise
`groq_max_concurrency` (subject to the Groq tier RPM/TPM ceiling) or
upgrade the Groq plan.

**Streamlit + auth.** `oauth2-proxy` sits in front of the Streamlit app
(per plan §155). The proxy injects `X-Forwarded-User` and
`X-Forwarded-Email` headers; Streamlit reads them via `st.context.headers`
(Streamlit ≥1.37). The Streamlit pages themselves do not validate auth
— that's the proxy's job.

---

## 12. API stability commitments

**v1 commitments:**

- Every response field has stable types (no widening / narrowing within v1).
- Every status code is stable for a given input class.
- Pagination semantics (`skip`, `limit`) are stable.
- The audit log (`experiment_events`) is append-only — entries are never
  modified or deleted.
- The construction-rule schema embedded in `latent_variables.construction_rule`
  is forward-compatible: new pattern types add new fields, never remove.
- Migration directives (§10) are stable: the parser format will not
  change within v1.

**Explicitly NOT committed in v1:**

- Exact prose generated by `ExplanationGenerator` (rule-based) — wording
  improvements ship without notice.
- Exact ordering of LVs returned by `GET /experiments/{id}/latent-variables`
  beyond the documented sort key (currently `created_at ASC`).
- Field order in JSON responses.
- Internal structure of `experiment_events.payload` JSONB — treat as
  opaque for cross-version queries.

---

## 13. Data egress inventory

This is the authoritative list of every field IVE may send to the
Groq / OpenAI-compatible LLM endpoint when `LLM_EXPLANATIONS_ENABLED=true`.

**Sent (when present and the LV/experiment passes the egress check):**

- Dataset name (sanitized of injection markers, truncated to 200 chars)
- Target column name **only when the target column is marked public**
- Latent variable's `name`, `description`
- LV statistics: `effect_size`, `presence_rate`, `stability_score`,
  `model_improvement_pct`, `effect_size_ci_lower/upper` (Phase B)
- LV `status` (validated / rejected / candidate)
- Names of columns referenced by the LV's construction rule, **only if
  every referenced column is marked `public`**
- Aggregated experiment-level counts: `n_findings`, `n_blocked_for_pii`,
  top-5 findings by importance score
- Pipeline status string (completed / failed / etc.)

**Never sent:**

- Raw row values from the dataset
- Individual record IDs
- Names of columns marked `non_public` (they don't appear in payloads at
  all — the egress check fires before the payload is built)
- API keys, secrets, environment variable values, file paths
- Any data from a dataset where every relevant column is `non_public`
  (such LVs are marked `disabled` with reason `pii_protection_per_column`)
- Any data when `LLM_EXPLANATIONS_ENABLED=false`

**Enforcement:** [tests/integration/test_sensitive_data_egress.py](../tests/integration/test_sensitive_data_egress.py)
runs on every PR and asserts that no non-public column name appears in
any captured prompt payload, that blocked LVs reach `disabled`, and that
flipping a column to `public` re-enables the LV's eligibility.

**Self-hosted alternative.** Setting `LLM_SELF_HOSTED_MODE=true` and
pointing `GROQ_BASE_URL` at a vLLM (or any OpenAI-compatible) endpoint
keeps every byte on-premises. See [§15](#15-self-hosted-llm-compatibility).

---

## 14. Sensitivity model

**Two-tier model** (per plan §174 + §203 — the binary is intentional;
column names themselves can be PII signals so a `name_only` middle tier
is unsafe by default):

| Sensitivity | LLM payload behavior | UI behavior |
|---|---|---|
| `public` | Column name + aggregated values may appear in LLM payloads | Visible in `affected-rows` API output |
| `non_public` (default) | Neither name nor values leave the perimeter; LVs referencing it are `disabled` with reason `pii_protection_per_column` | Hidden from `affected-rows` output |

**Default:** every column is `non_public` on upload. The user opts in by
promoting columns to `public` via:
- The Streamlit Dataset Settings page (PR-7)
- `PUT /api/v1/datasets/{dataset_id}/columns/`

**No middle tier.** `name_only` was rejected in design review because
column names like `hiv_positive`, `ssn_last_four`, `annual_household_income_bracket`
are themselves identifying signals. Treating column names as inherently
lower-sensitivity than values would be wrong; the binary keeps the
contract simple and safe.

**Audit trail.** The `dataset_column_metadata` table records every
sensitivity decision with `created_at` / `updated_at` timestamps. The
`auth_audit_log` table (PR-2) records who set each.

**Cascading effect.** When a column flips from `non_public` to `public`,
existing LVs that referenced it remain `disabled` until the next
experiment runs (per plan §40 — the admin endpoint
`POST /experiments/{id}/regenerate-llm-explanations` is the way to force
re-evaluation; planned, not yet shipped).

---

## 15. Self-hosted-LLM compatibility

`LLM_SELF_HOSTED_MODE=true` + `GROQ_BASE_URL=<your-endpoint>` switches
the LLM client from Groq's hosted API to any OpenAI-compatible server.

**Tier 1 (highest fidelity):** `llama-3.3-70b-instruct` on 4× A100 (80 GB)
or 2× H100 — output qualitatively matches Groq output. Validator profile
(`groq_llama_3_3_70b`) transfers without retuning.

**Tier 2 (mid):** `llama-3.1-8b-instruct` on a single A100/L40S — usable,
but the validator profile must be re-tuned for the model's failure modes.
`LLM_VALIDATOR_PROFILE=vllm_llama_3_1_8b` (planned).

**Tier 3 (CPU-only):** out of scope. CPU inference of a 70B model is
multi-second-per-token; unsuitable for our latency budget.

**Stack recommendation:** vLLM behind nginx with TLS, Redis-backed
session caching. Detailed setup in `docs/self_hosted_llm.md` (planned).

**Egress guarantee in self-hosted mode.** All §13 invariants still hold —
the only thing that changes is the network destination. `LLM_SELF_HOSTED_MODE=true`
does **not** weaken the per-column sensitivity check; non-public columns
still never leave the IVE process boundary even when the LLM is on the
same Kubernetes namespace.

---

## 16. Selective-inference methodology

K-fold cross-fitting (`K=5`, configurable `selective_inference_k`) protects
subgroup discovery against selection bias:

1. Split data into K folds.
2. For each fold *k*: train base models + run subgroup/cluster/interaction
   discovery on folds ≠ *k*; apply the discovered definitions to fold *k*;
   compute effect size and BCa CI on fold *k* only.
3. Aggregate: effect size = mean of K per-fold estimates; CI = 2.5/97.5
   percentiles of the union of K bootstrap distributions.
4. Subgroup must be discovered in ≥3 of K splits to be reported. Subgroups
   discovered in only 1–2 splits get `status='rejected'`,
   `rejection_reason='cross_fit_unstable'`.

**Schema fields:** `error_patterns.cross_fit_splits_supporting`,
`error_patterns.selection_corrected: bool`.

**Threshold for enablement:** `n_rows >= 1500`. Default
`selective_inference_enabled=True` for `n_rows < 50_000`; users opt in
above (5× modeling cost trade-off).

**Anti-conservative caveat.** The ≥3-of-K stability vote is itself a
selection step that we do *not* correct for. Empirical CI-width
underestimation on simulated data: 2–4%. Users requiring fully-corrected
CIs set `selective_inference_strict=True` (Phase B.5): 3-of-K folds
vote on stability; the remaining 2-of-K folds compute CIs.

## 17. Operational posture

- **Database:** Postgres 16. Production runs on managed Postgres with
  daily snapshots + 7-day PITR.
  - RPO: 5 min (PITR granularity). RTO: 30 min (provider-dependent).
  - Self-hosted: `pg_dump` nightly to S3 + WAL archiving. See
    `docs/runbooks/postgres_backup.md`.
- **Redis:** broker (db 0) / results (db 1) / LLM cache (db 2). Cache is
  regeneratable; broker/results retained per Celery `result_expires`.
- **All user-generated content** (annotations, feedback, audit logs) is
  included in backups.

## 18. Cost model (Groq pricing as of 2026-04-27)

| Daily volume | Estimated cost | Constraint |
|---|---|---|
| 1,000 experiments | $20 / day = $7.3k / year | Negligible |
| 10,000 experiments | $200 / day = $73k / year | `llm_daily_token_budget` floor |
| 100,000 experiments | $730k / year | Groq paid-tier RPM/TPM ceilings, not cost |

`llm_daily_token_budget` enforces a daily kill switch (circuit-break for
the rest of the UTC day past the budget). Cache hit rate of ~30%
(estimated from typical re-run patterns) reduces effective cost by ~30%.

Self-hosted (Tier 1: 4×A100 + vLLM serving llama-3.3-70b) capex/opex is
out of scope for v1; documented in `docs/self_hosted_llm.md` (Phase D).

## 19. Data-lineage tracking

`dataset_column_versions` records `(dataset_id, column_name, dtype,
value_hash, version)`. Hashes computed asynchronously after upload via the
`compute_dataset_lineage` Celery task. Lineage status surfaces as
`lineage_status='pending|ready'` on the dataset detail page.

**Lineage detection rules** (current vs prior version):

| Event | Heuristic |
|---|---|
| `retype` | Same column name, different dtype |
| `value_change` | Same name + dtype, different value_hash |
| `rename_candidate` | Different name, same dtype, value_hash within Hamming distance ≤2 (requires user confirmation) |
| `drop` | Column in v_old, absent in v_new with no rename match |
| `add` | Column in v_new, absent in v_old |

LVs whose `derived_features.source_columns` reference any column with a
`retype` / `value_change` / `drop` / `rename_candidate` event get
`apply_compatibility='requires_review'`. Bulk operator action via
`make regenerate-disabled-explanations EXPERIMENT_ID=...`.

**`model_version` opacity.** Groq publishes model identifiers but not
weight versions. Provider-side weight or quantization updates under the
same identifier are detected only by the weekly behavioral drift suite
(plan §164) with up to 7-day detection latency. The daily smoke (5
canonical prompts, <1 min) reduces severe-regression detection to ≤24 h.

## 20. Multi-tenancy

**v1 is single-tenant per deployment.** Each API key is a user identity
but **not** a tenant boundary. Row-level access between users is
application-discipline only — not Postgres RLS, not ORM-enforced filters.

Multi-tenant deployment requires Phase D scope:
- Row-scoped queries
- `tenant_id` FK on every user-data table
- Postgres RLS policies

Single-deployment-per-customer is the recommended deployment topology
until Phase D ships.

## 21. Release-train cadence

- `main` is always shippable (CI-green, all migrations applied).
- Production deploys are operator-triggered weekly (Tuesdays 10:00 UTC).
- Hotfix path: PRs labeled `hotfix` deployable off-cadence after a single
  approval and full CI run.
- Deprecation timers in §6 measured in calendar weeks, not deploys.

## 22. Multiclass classification support

Multiclass models use `XGBClassifier(objective="multi:softprob")`. The
pipeline supports:

| Feature | Multiclass support |
|---|---|
| Trained models per fold | ✓ |
| Phase 5 holdout uplift (log-loss metric) | ✓ |
| HPO over multiclass search space | ✓ |
| Stacked ensemble with multinomial meta-learner | ✓ |
| LV apply for non-residual-derived LVs | ✓ |
| Subgroup discovery (KS on residuals) | ✗ (logs `multiclass_residual_detection_unsupported`) |
| HDBSCAN clustering on residuals | ✗ |
| SHAP interaction mining | ✗ |
| Variance regime (B8) | ✗ |
| B4 effect-size CIs | ✗ |

`ExperimentResponse.multiclass_supported_features: list[str]` makes the
support matrix machine-readable.

## 23. `holdout_uplift` versioning

`holdout_uplift` semantics shifted post-B3 (linear baseline → ensemble
baseline). To avoid silently changing client-visible numbers:

| Field | Computed against | Status |
|---|---|---|
| `holdout_uplift` | Linear baseline + LV vs linear baseline | Pre-B3 contract; populated indefinitely for backwards compatibility |
| `final_holdout_uplift` | Ensemble + LV vs ensemble (untouched final-holdout split) | Canonical post-B3; new clients should read this |
| `selection_holdout_uplift` | Ensemble + LV vs ensemble (selection_holdout half used by greedy selection) | Diagnostic only — never the headline number |

Responses carry `Deprecation: true` and `Link: <docs URL>; rel="deprecation"`
when only `holdout_uplift` is consumed. Sunset announced ≥6 months per §6.

## 24. CI gates

| Gate | What it checks | Workflow / target |
|---|---|---|
| Lint | ruff + mypy strict | `make lint` |
| Unit | `tests/unit/` all green | `.github/workflows/ci.yml` |
| Integration | `tests/integration/` | same |
| Statistical | `tests/statistical/` (FPR, CI coverage, recall) | `make test-statistical` |
| Contract drift | RESPONSE_CONTRACT.md changes when config / schema / migrations change | `.github/workflows/contract_drift.yml` |
| Migration directive | Every alembic migration has a `deploy:` line | `.github/workflows/migrations_check.yml` |
| Single alembic head | `alembic heads` returns one line | startup preflight + CI |
| Demo-dataset hashes | `tests/fixtures/demo_dataset_manifest.json` matches | `make test-statistical` |
| Benchmark | P95 phase duration not >25% over baseline (median-of-5 on self-hosted runner) | `.github/workflows/benchmark.yml` |
| Sensitive-data egress | Groq mock client sees zero `non_public` column names/values | `tests/e2e/test_sensitive_data_egress.py` |
| Adversarial validator | All 20 corpus cases blocked by `composite_validate` | CI gate on `src/ive/llm/` PRs |
| CODEOWNERS consistency | Every directory in CODEOWNERS matches plan §121 | `tests/test_codeowners_consistency.py` |

---

## Appendix: change log

| Date | Version | Change |
|---|---|---|
| 2026-04-28 | v1 (PR-9) | Initial publication. Phase A scope. Sections 1-15. |
| 2026-04-28 | v1.1 (Wave 1) | Added §16-§24 (selective inference, ops posture, cost model, lineage, multi-tenancy, release cadence, multiclass support, holdout_uplift versioning, CI gates). |

---

*Owner: ML lead. Cross-reference parity with code is gated by
`docs/RESPONSE_CONTRACT.md` change-required CI check (per plan §122) once
the gate is wired up in the CI workflow. Send corrections via PR.*
