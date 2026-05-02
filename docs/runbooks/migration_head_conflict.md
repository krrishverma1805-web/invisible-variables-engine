# Runbook — Alembic Multi-Head Conflict

**Trigger:** API/worker startup fails with `Multiple heads detected; run
alembic merge`. CI gate `alembic-heads-single` fails on PR.
**Severity:** critical (blocks deploys)
**Owner:** backend lead

## Symptom

```
$ alembic heads
abc123 (head)
def456 (head)
```

Two parallel migration branches landed on `main` from independent PRs;
neither is a strict ancestor of the other. The startup preflight in
`docker-compose.yml` refuses to boot the API service.

## Diagnosis

```bash
# Confirm both heads exist
PYTHONPATH=src poetry run alembic heads

# Inspect the branch point
PYTHONPATH=src poetry run alembic branches

# What's the latest common ancestor?
PYTHONPATH=src poetry run alembic history | head -40
```

## Mitigation

Create a merge migration that names both heads as parents:

```bash
PYTHONPATH=src poetry run alembic merge -m "merge_<short_description>" \
  abc123 def456
```

This generates a no-op migration with both heads as `down_revision`.
**Requirements before merging the resulting file:**

- [ ] Docstring includes `deploy: same-release` directive (the merge is
      atomic by definition, so this is the only valid choice).
- [ ] Both parent migration directives still apply individually
      (a merge cannot silently relax a `schema-first` requirement).
- [ ] Both parents' `upgrade()` chains were tested individually before
      this merge — if not, re-run `alembic upgrade head` against a
      throwaway database to catch ordering bugs.
- [ ] The CI gate now shows a single head.

```bash
# Smoke test against an empty DB
docker-compose up -d postgres
PYTHONPATH=src poetry run alembic upgrade head
PYTHONPATH=src poetry run alembic downgrade base
PYTHONPATH=src poetry run alembic upgrade head
```

## Prevention

- Each PR description carries a `Depends-On:` field listing referenced
  PR numbers (plan §120). CI workflow refuses to merge until upstream
  PRs land.
- Authors of new migrations rebase their branch onto `main` immediately
  before merging, regenerate the `down_revision` if needed, and verify
  `alembic heads | wc -l` still returns 1.

## Postmortem checklist

- [ ] Why did two migrations land in parallel without `Depends-On:`?
- [ ] Did the CI gate actually run on both PRs (or was one excluded
      by path filters)?
- [ ] Does the merge migration's `down_revision` tuple list both
      parents in the order the operator expects?
