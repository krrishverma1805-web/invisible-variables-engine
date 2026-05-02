# Runbook — LLM Token Budget Exhausted

**Alert:** `ive_llm_daily_token_budget_burnrate`
**Severity:** warning (escalates if ignored)
**Owner:** ml-platform

## Symptom

Counter `ive_llm_tokens_total{kind="total"}` is incrementing >100k
tokens/hour for 30 minutes. At this rate the daily ceiling
(`llm_daily_token_budget`, default 1M) will exhaust before UTC rollover;
when it does, every new generation routes to fallback with reason
`token_cap_daily`.

## Likely cause (rank-ordered)

1. **Runaway batch of large experiments** — a user uploaded 10× the
   normal dataset and the LV count exploded.
2. **Cache hit-rate collapse** — `ive_llm_cache_total{outcome="hit"}`
   went to ~0 because of a `LLM_PROMPT_VERSION` bump (cache
   invalidation) or a Redis flush.
3. **Per-experiment cap not enforced** — `llm_per_experiment_token_cap`
   logic broke; one experiment is consuming the entire budget.
4. **Genuine growth** — sustained increase in real usage; budget needs
   raising.

## Diagnosis

```bash
# Top experiments by recent token spend
psql "$DATABASE_URL" -c \
  "SELECT id, dataset_id,
          (SELECT count(*) FROM latent_variables
           WHERE experiment_id = e.id) as n_lvs,
          extract(epoch from now() - created_at) / 60 as age_min
   FROM experiments e
   ORDER BY created_at DESC LIMIT 20;"

# Cache hit rate (Prometheus)
sum(rate(ive_llm_cache_total{outcome="hit"}[1h])) /
clamp_min(sum(rate(ive_llm_cache_total[1h])), 1)

# Recent token spend by function
sum by (function) (rate(ive_llm_tokens_total{kind="total"}[15m]))
```

## Mitigation

| Cause | Action |
|---|---|
| Runaway experiment | Cancel via `POST /api/v1/experiments/{id}/cancel`; cancellation cascades to in-flight LLM batches per plan §171. |
| Cache invalidation aftermath | Wait for cache to re-warm (typically 30–60 min). Consider lowering `llm_batch_size_lvs` temporarily to share work across experiments. |
| Per-experiment cap broken | Restart workers; check `experiments.llm_tokens_consumed` column matches the Prometheus counter. File a bug. |
| Genuine growth | Raise `llm_daily_token_budget` in production env; document the new ceiling in §15 of `docs/RESPONSE_CONTRACT.md`. |

**Emergency stop:**
```bash
LLM_EXPLANATIONS_ENABLED=false  # workers will mark all new rows 'disabled'
```

## Postmortem checklist

- [ ] Did the cap actually cut off the runaway experiment, or did it
      finish over-budget?
- [ ] Did `ive_llm_fallback_total{reason="token_cap_*"}` correlate
      with the spike?
- [ ] Was the predictive cap (§108) firing or only the reactive one?
      The reactive one is a bug — predictive should preempt.
- [ ] Update §15 cost-model table in `docs/RESPONSE_CONTRACT.md` if the
      budget needs to be permanently higher.
