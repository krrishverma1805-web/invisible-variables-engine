# Runbook — Postgres Backup + Restore

**Trigger:** disaster recovery, planned schema rollback, or scheduled
restore drill.
**Severity:** procedure (no alert)
**Owner:** platform

## Backup posture

| Deployment | Strategy | RPO | RTO |
|---|---|---|---|
| Managed (RDS / Cloud SQL / Aiven) | Daily snapshot + 7-day PITR | 5 min | 30 min (provider-dependent) |
| Self-hosted | `pg_dump` nightly + WAL archiving | 5 min | 1 h (manual restore) |

All user-generated content (annotations, feedback, audit logs) included.

## Self-hosted backup procedure

```bash
# Nightly cron — runs as the postgres user.
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
pg_dump -Fc -Z 6 -f "/backups/ive-${TS}.dump" "$DATABASE_URL"
aws s3 cp "/backups/ive-${TS}.dump" "s3://ive-backups/${TS}/full.dump"

# Cleanup local dumps >7 days
find /backups -name 'ive-*.dump' -mtime +7 -delete
```

WAL archiving via `archive_command` in `postgresql.conf`:

```
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://ive-backups/wal/%f'
restore_command = 'aws s3 cp s3://ive-backups/wal/%f %p'
```

## Restore procedure

### Full restore from latest dump

```bash
# Stop the API + worker so writes don't fight the restore
docker-compose stop api worker

# Restore into a fresh database
createdb ive_restored
pg_restore --clean --if-exists -d ive_restored \
  -j 4 ive-2026-04-28T02-00-00.dump

# Atomic swap (re-point DATABASE_URL or rename DBs)
psql -c "ALTER DATABASE ive RENAME TO ive_quarantined;"
psql -c "ALTER DATABASE ive_restored RENAME TO ive;"

# Bring services back up
docker-compose up -d api worker
```

### PITR (managed Postgres)

Use the provider's console:

* RDS: "Restore to point in time" — pick a target timestamp within the
  retention window.
* Cloud SQL: `gcloud sql backups restore` with `--backup-instance` and
  `--restore-time`.
* Aiven: per their docs.

Resulting instance is a clone; re-point `DATABASE_URL` after smoke-test.

### Smoke tests after restore

```bash
# 1. Schema match
PYTHONPATH=src poetry run alembic current

# 2. Row counts roughly match expectation
psql "$DATABASE_URL" -c "
SELECT 'experiments' as tbl, count(*) FROM experiments
UNION ALL SELECT 'datasets', count(*) FROM datasets
UNION ALL SELECT 'latent_variables', count(*) FROM latent_variables
UNION ALL SELECT 'share_tokens', count(*) FROM share_tokens
UNION ALL SELECT 'auth_audit_log', count(*) FROM auth_audit_log;"

# 3. Run the e2e test against the restored DB
PYTHONPATH=src poetry run pytest tests/integration/ -q
```

## Restore drill schedule

Quarterly (last Thursday of Mar / Jun / Sep / Dec). Drill targets:

- Restore the previous-quarter snapshot to a staging DB.
- Run the integration test suite against the restored DB.
- Measure actual RTO; if >2× SLO, escalate.

## Postmortem checklist

- [ ] Did the restore complete within the documented RTO?
- [ ] Were the smoke tests sufficient to declare the restore healthy?
- [ ] Was an audit-log entry written for the restore action (who, when,
      why)?
- [ ] Update this runbook if the procedure changed.
