# Competitive Baseline — IVE

**Effective date:** 2026-04-28
**Owner:** ML lead
**Plan reference:** §92 + §143

This document names the comparison set IVE benchmarks against, the five
axes on which we intend to win, and the **falsifiable measurement
protocol** for each. The point is that "best-in-class" must be a number
the team can lose at — otherwise it's marketing, not engineering.

## Reference platforms

We compare against these because they ship at production scale and
overlap meaningfully with IVE's domain:

| Platform | Overlap with IVE | Why benchmarked |
|---|---|---|
| **DataRobot** | AutoML + explanation generation | Closest commercial overlap; "natural-language insights" market. |
| **H2O Driverless AI** | AutoML + interpretability | Strong on feature engineering + SHAP surfacing. |
| **Dataiku** | End-to-end DS with explanation modules | Mid-market player; weaker on causal but strong on UX. |
| **Vertex AI Experiment Tracking** | Reproducibility + serving lineage | Cloud-native baseline for reproducibility claims. |

## The 5 axes

### Axis 1 — Honest causal rigor

**Claim:** IVE flags confounded effects via DML and refuses to report
unexamined causal claims.

**Measurement:**
- Synthetic dataset `confounded_by_construction.csv` (committed).
- Run IVE with `causal_checker` enabled.
- Assertion: ≥90% of seeds (n=20) produce
  `causal_warnings` containing `confounded_by_dml` for the
  confounded-by-construction LV.
- Comparison platforms: manually inspected for whether they distinguish
  confounded vs causal at all (most do not).

**Test fixture:** `tests/statistical/test_causal_dml_recall.py`
(planned).

### Axis 2 — Latent-variable explanations as a first-class output

**Claim:** IVE produces hedged, statistically grounded prose
explaining each LV — not just feature importance.

**Measurement (revised per plan §189):**
- N=50 raters (power calc supports detecting 0.3 effect-size differences
  at p<0.05).
- Within-subjects paired comparison; each rater sees both versions.
- Bonferroni-corrected α across 3 dimensions: clarity, actionability,
  hedging-appropriateness (effective threshold p<0.0167 per dimension).
- Prompt selection: an independent third party (not IVE authors) picks
  the LV/dataset pairs from a stratified random sample of 30 finished
  experiments.
- Claim downgraded to: *"In-house preference signal — IVE wins on ≥2 of
  3 dimensions at Bonferroni-corrected p<0.05; not a publishable
  head-to-head benchmark."*

**Test fixture:** Reader-study harness (planned, post-Phase A).

### Axis 3 — FPR-calibrated stability

**Claim:** IVE publishes empirical FPR guarantees; stability thresholds
are calibrated against synthetic noise, not heuristic.

**Measurement:**
- Run experiment on `demo_datasets/no_hidden_random_noise.csv` × 200
  seeds.
- FPR upper-95% Clopper-Pearson CI ≤ 7%.
- No competitor publishes empirical FPR — this is a defensible
  technical claim, not a head-to-head comparison.

**Test fixture:** `tests/statistical/test_stability_fpr.py` + nightly
`fpr_sentinel_run` Celery beat task.

### Axis 4 — Hedged business-readable narratives

**Claim:** Groq-enhanced narratives never invent statistics, never use
causal verbs without hedge markers, and never reference fields not in
the input.

**Measurement:**
- Validator pass-rate on a held-out behavioral test corpus
  (`tests/llm/behavioral_corpus.json`, 50 cases) ≥ 95%.
- No hallucinated stats: 0 cases in the corpus.
- No banned phrases without hedge: 0 cases.
- Quarterly re-run with current Groq weights; alert on >5% pass-rate
  drift (plan §164).

**Test fixture:** `tests/llm/test_behavioral_corpus.py` (planned).

### Axis 5 — One-command reproducibility

**Claim:** Same `(random_seed, study_seed, cv_seed)` → byte-identical
output across DB columns.

**Measurement:**
- `make test-reproducibility` runs the same experiment twice with the
  full reproducibility tuple fixed.
- Diffs every persisted column. Threshold: zero non-trivial diffs
  (timestamps and UUIDs excluded, all numeric / structural columns
  byte-identical).

**Test fixture:** `tests/integration/test_reproducibility.py` (planned).
CI gate.

## What we explicitly do NOT claim

| Non-claim | Rationale |
|---|---|
| "Best AutoML" | Not the goal — IVE is a residual-discovery tool, not a model-builder. |
| "Causally identifies hidden variables" | DML is a screen for confounding, not proof of causation. We hedge and we mean it. |
| "Faster than {DataRobot,H2O,Dataiku}" | Different problem shape; they finish faster on their problems and slower on ours. |
| "Production-grade for >10M-row datasets in v1" | Plan §51 explicitly accepts the limit. Phase D scoping required for true hyper-scale. |

## Comparison protocol

Re-run quarterly:

1. Spin up free-tier accounts on each comparison platform.
2. Upload `delivery_hidden_weather.csv`, `customer_churn_hidden_signal.csv`,
   `confounded_by_construction.csv`.
3. For Axis 1: capture each platform's response to the confounded
   dataset; record screenshots in `docs/competitive_baseline_screenshots/`.
4. For Axis 2: run the reader study (50 raters × 30 stratified pairs).
5. Axes 3-5: run our own measurement scripts; competitors don't publish
   matching numbers.
6. Update the comparison table below with quarterly readings.

## Latest readings (placeholder — updated quarterly)

| Quarter | Axis 1 confound recall | Axis 2 reader pref (3 dim, p) | Axis 3 FPR upper-95 | Axis 4 validator pass | Axis 5 byte-identical |
|---|---|---|---|---|---|
| 2026-Q2 | TBD | TBD | TBD | TBD | TBD |

## Document changelog

| Date | Change |
|---|---|
| 2026-04-28 | Initial draft; targets defined, fixtures named (some planned). |
