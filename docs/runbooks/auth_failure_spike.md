# Runbook — Auth Failure Spike

**Alert:** `ive_auth_failure_spike`
**Severity:** warning
**Owner:** platform

## Symptom

`sum(rate(starlette_requests_total{status=~"401|403"}[5m])) > 1` for ≥5
minutes. Multiple clients are getting 401/403 responses.

## Likely cause (rank-ordered)

1. **Compromised key** — a key leaked, attacker is probing other keys
   off it.
2. **Recently-rotated key** — a deployment rotated `X-API-Key` and not
   all clients picked up the new value.
3. **Scope mismatch** — a new endpoint was gated on `Scope.ADMIN` but
   the client's key has only `read`/`write`.
4. **Configuration drift** — `_EXEMPT_PREFIXES` removed something that
   should be exempt (e.g. `/api/v1/health`).
5. **Hot revoke** — operator revoked a key without notifying the
   client.

## Diagnosis

```bash
# Top failing keys (last hour)
psql "$DATABASE_URL" -c \
  "SELECT key_name_hint, count(*) as failures, max(created_at) as last_seen
   FROM auth_audit_log
   WHERE outcome = 'fail' AND created_at > now() - interval '1 hour'
   GROUP BY key_name_hint
   ORDER BY failures DESC LIMIT 10;"

# Distribution of failure reasons
psql "$DATABASE_URL" -c \
  "SELECT failure_reason, count(*)
   FROM auth_audit_log
   WHERE outcome = 'fail' AND created_at > now() - interval '1 hour'
   GROUP BY failure_reason ORDER BY count(*) DESC;"

# Source IP distribution (looking for single-source attack)
psql "$DATABASE_URL" -c \
  "SELECT client_ip, count(*) as failures
   FROM auth_audit_log
   WHERE outcome = 'fail' AND created_at > now() - interval '1 hour'
   GROUP BY client_ip ORDER BY failures DESC LIMIT 10;"
```

## Mitigation

| Cause | Action |
|---|---|
| Compromised key | Revoke immediately: `DELETE /api/v1/api-keys/{key_id}` (admin scope). Issue replacement out-of-band. |
| Single-source brute force | Block the IP at the firewall / load balancer. Lower per-IP rate limit if the key is unknown. |
| Rotation lag | Notify clients; consider extending the deprecated-key grace period (set both old + new keys active for 24h then revoke old). |
| Scope mismatch | Issue a new key with the correct scope. Don't widen scope on existing key — issue a new one with the right scopes per least-privilege. |
| Misconfig | Restore the `_EXEMPT_PREFIXES` entry in `src/ive/api/middleware/auth.py`. |

## Postmortem checklist

- [ ] Was the key rotation procedure followed (overlap window
      documented, all client owners notified)?
- [ ] Did the auth audit log retain enough detail (failure_reason,
      client_ip, user_agent) to diagnose without forensics?
- [ ] If brute-forced: do we need lower per-IP rate limits as the
      default?
- [ ] If misconfig: add an integration test that asserts every public
      endpoint returns a non-401 with no key.
