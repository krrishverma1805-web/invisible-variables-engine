# Runbook — Groq Circuit Breaker Open

**Alert:** `ive_llm_circuit_breaker_open` (also fired by `ive_llm_fallback_rate_high`)
**Severity:** critical
**Owner:** ml-platform

## Symptom

Prometheus gauge `ive_llm_circuit_breaker_state{service="groq"}` reads 1
for ≥5 minutes. Endpoints return rule-based explanations with
`explanation_source="rule_based"`; no new `llm_explanation_status='ready'`
rows are being written.

## Likely cause (rank-ordered)

1. **Groq upstream outage** — check status.groq.com.
2. **Expired/revoked `GROQ_API_KEY`** — confirm via `curl` with the
   current key.
3. **Network egress blocked** — VPC firewall, NAT exhaustion, DNS.
4. **Bad prompt template** producing `LLMBadRequest` (400) — though 4xx
   should NOT count toward the breaker per plan §109. If it does, the
   breaker classifier has regressed.
5. **Daily token budget exhausted** (`ive_llm_daily_token_budget_burnrate`
   would have alerted earlier).

## Diagnosis

```bash
# 1. Confirm the breaker really is open in Redis (db 2 by default)
redis-cli -n 2 GET ive:llm:breaker:groq:open

# 2. Failure counter snapshot
redis-cli -n 2 GET ive:llm:breaker:groq:fail

# 3. Recent client errors in structlog
journalctl -u ive-worker -n 200 | grep llm.client

# 4. Direct Groq probe
curl -sS -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models | jq '.data[0]'
```

## Mitigation

| Cause | Action |
|---|---|
| Groq outage | Wait it out. Cooldown is `groq_circuit_breaker_cooldown_seconds` (default 300s); breaker auto-closes after a successful probe. |
| Expired key | Rotate key (see `docs/runbooks/auth_failure_spike.md`); restart workers. |
| Bad prompt | Revert the offending PR; bump `LLM_PROMPT_VERSION` to invalidate cache. |
| Want to force-close manually | `redis-cli -n 2 DEL ive:llm:breaker:groq:open ive:llm:breaker:groq:fail` |
| Want to disable LLM entirely | `LLM_EXPLANATIONS_ENABLED=false` env-flip + worker restart. |

## Postmortem checklist

- [ ] Was 5xx / 429 categorisation correct (only those count toward breaker)?
- [ ] Did fallback prose match rule-based output character-for-character
      against pre-incident baselines?
- [ ] Did any experiment finish with `llm_explanation_status='failed'`
      that should have been `'disabled'`?
- [ ] Update `groq_circuit_breaker_threshold` if the threshold (5
      consecutive failures) is too tight or too loose for the observed
      noise floor.
