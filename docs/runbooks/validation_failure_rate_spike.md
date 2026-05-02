# Runbook — LLM Validation Failure Rate Spike

**Alert:** `ive_llm_validation_failure_rate_high`
**Severity:** warning
**Owner:** ml-platform

## Symptom

`sum(rate(ive_llm_validation_failed_total[10m])) > 0.5/sec` sustained
≥10 minutes. LLM outputs are being rejected by `composite_validate`
faster than normal; the system is falling back to rule-based prose for
those LVs.

## Likely cause (rank-ordered)

1. **Recent prompt change** — A `LLM_PROMPT_VERSION` bump or template
   refactor produces output the validator rejects.
2. **Upstream model drift** — Groq pushed a weight update under the same
   `model_id`. Validator profile (`src/ive/llm/profiles/<id>.py`) is now
   tuned to a slightly different model.
3. **Per-experiment payload regression** — A new field added to the
   payload schema changes the prompt structure (B4's `effect_size_ci`,
   for example) and the model started inventing numbers.
4. **Adversarial input** — User-controlled column / segment names
   contain text that breaks the prompt past `sanitize_user_input()`.

## Diagnosis

```bash
# Bucket failures by reason (Prometheus query)
sum by (reason) (rate(ive_llm_validation_failed_total[15m]))

# Sample 10 recent failures with full payloads
psql "$DATABASE_URL" -c \
  "SELECT lv_id, llm_explanation, llm_explanation_status, rejection_reason
   FROM latent_variables
   WHERE llm_explanation_status='failed'
     AND llm_explanation_generated_at > now() - interval '30 minutes'
   ORDER BY llm_explanation_generated_at DESC LIMIT 10;"

# Re-run a failing prompt against current Groq + see what comes back
PYTHONPATH=src poetry run python scripts/replay_failed_prompt.py <lv_id>
```

## Mitigation

| Top failure reason | Action |
|---|---|
| `numeric_grounding` | Recent prompt change leaking a derived fact? Inspect prompt diff. Check that `canonicalize_numbers()` covers any new number forms. |
| `banned_phrase` | Model regressing on causal-verb hedging. Check if upstream model was updated; flip to fallback by setting `LLM_EXPLANATIONS_ENABLED=false` while you retune. |
| `field_grounding` | Payload field renamed without the synonym set being updated. |
| `prompt_injection` | Sanitization missed something — find the offending input, extend `sanitize_user_input()`, add to adversarial corpus. |

**Rollback path:** revert the offending PR (prompt change is the most
common cause); the cache key includes `prompt_template_sha[:16]` so
reverting auto-invalidates the bad-prompt cache entries.

## Postmortem checklist

- [ ] Was the offending change behind a feature flag? If not, why not?
- [ ] Did `tests/unit/test_llm_prompts.py` golden-snapshot tests catch
      the prompt drift? Why didn't they?
- [ ] Did the behavioral drift suite (weekly) catch model-side drift?
      Was its alarm threshold too high?
- [ ] Add the offending output text to `tests/unit/test_llm_validators.py`
      as a regression case.
