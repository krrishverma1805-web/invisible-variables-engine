# Model Selection

**Effective date:** 2026-04-29
**Owner:** ML lead
**Plan reference:** §93

This document records the why behind IVE's pinned LLM model and the
contract for changing it. The cache key includes both `LLM_PROMPT_VERSION`
and the prompt-template SHA (RC §1, §10), so a model bump is at most a
two-line config change — but the trade-offs deserve to be deliberate.

## Pinned model

| Setting | Value | `pinned_at` |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | 2026-04-27 |
| `LLM_VALIDATOR_PROFILE` | `groq_llama_3_3_70b` | 2026-04-27 |

## Rationale

- **Quality / speed / cost balance.** As of 2026-04-27, this is the best
  Groq-served model for analytical writing. 70B parameters carry enough
  nuance to produce hedged, statistic-grounded prose; smaller (8B) is
  too thin, larger (405B) provides no measurable quality lift on our
  validator gate at ~5–10× the cost.
- **OpenAI-compatible.** Reuses the existing `httpx` async client; no
  bespoke client.
- **JSON mode supported.** Plan §103's batched-LV prompt
  (`lv_explanation_batch v1`) requires JSON mode; this model exposes it.
- **Free-tier friendly.** Acceptable RPM/TPM ceilings for development;
  paid tier scales linearly past the IVE per-deployment cost estimates
  in `docs/RESPONSE_CONTRACT.md §15`.

## Why not the alternatives

### `llama-3.1-8b-instant` / similar small models

- Consistently fails numeric grounding more often (~15% vs ~5%).
- Less reliable hedging — uses causal verbs without hedge markers
  more often, increasing fallback rate.
- Used as the Tier-2 self-hosted option (`docs/self_hosted_llm.md`)
  with a relaxed validator profile, not as the default.

### `llama-3.3-405b` / larger frontier models

- ~5–10× cost on Groq.
- No measurable quality lift on the validator pass-rate metric.
- Larger latency variance.
- Not justified by the use case (analytical paraphrase, not creative
  writing or complex reasoning).

### Anthropic / OpenAI hosted models

- Equivalent quality, **3–10× cost** on per-token billing (we use Groq
  for cost-per-token economics).
- Use them when the Groq dependency is unacceptable and a self-hosted
  Tier 1 is also unavailable. Add a `LLM_VALIDATOR_PROFILE` for them
  before promoting.

## Pinning surfaces

The pin lives in three places that must stay in lockstep:

1. `[.env.example](../.env.example)` — `GROQ_MODEL=llama-3.3-70b-versatile`
2. `[model_capabilities.yaml](../model_capabilities.yaml)` — `groq:` section
3. This doc — `pinned_at` field above

CI gate `contract_drift_check` (RC §22) flags any change to (1)/(2)
that doesn't update this doc.

## Migration runbook

When Groq announces a deprecation or a better model becomes available:

1. **Freeze.** Don't bump `LLM_PROMPT_VERSION` during the migration
   window — that's a separate concern.
2. **Behavioral suite.** Run the weekly behavioral suite (50 cases) +
   adversarial corpus (20 cases) against the candidate model:
   ```bash
   GROQ_MODEL=<candidate> poetry run pytest tests/unit/test_behavioral_corpus.py
   GROQ_MODEL=<candidate> poetry run pytest tests/unit/test_adversarial_corpus.py
   ```
   Pass rate must match or exceed current model within 2 percentage
   points.
3. **Profile.** Add `src/ive/llm/profiles/<candidate_id>.py` with
   model-specific banned phrases, tolerance bands, and recommended
   `max_tokens`. (Plan §141 schema.)
4. **A/B comparison.** With both profiles available, A/B compare on
   production cache misses for 1 week. Track:
   - Validator pass rate per model.
   - Fallback rate per model.
   - Reader-rated clarity (smaller sample, ~20 paired comparisons).
5. **Promote.** Bump `GROQ_MODEL`, `LLM_VALIDATOR_PROFILE`, and
   `LLM_PROMPT_VERSION` (auto-invalidates cache via `prompt_template_sha[:16]`
   in cache key). Update `pinned_at` in this doc.
6. **Tag the release** with `[model-migration]`. Notify operators via
   the changelog.

## Drift detection

Two scheduled jobs guard against silent regressions:

| Job | Cadence | What it does | Alert threshold |
|---|---|---|---|
| Daily smoke | Daily, ~02:00 UTC | 5 canonical prompts against live Groq | Any fail → page |
| Weekly drift suite | Weekly, Sun 03:00 UTC | 50-prompt corpus against live Groq | Pass rate <95% → ticket |

Both alerts route via the existing Prometheus rules
(`ops/prometheus/alerts.yaml`) → Sentry → on-call.

## Provider-side weight updates

Groq publishes model identifiers but **not** underlying weight versions.
A silent quantization or fine-tuning update under the same identifier
is detected only by the drift suite, with up to 7-day latency.
Documented in RC §19 ("`model_version` opacity").

## What this doc IS and IS NOT

| Is | Is not |
|---|---|
| The why behind today's pin | A list of all models we ever tested |
| The migration contract | A model comparison benchmark |
| The drift watchdog spec | A vendor selection checklist (use `docs/competitive_baseline.md` for that) |
