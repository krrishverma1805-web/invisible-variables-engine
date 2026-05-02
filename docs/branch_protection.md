# Branch Protection — IVE

**Owner:** security-reviewer + platform-lead
**Plan reference:** §121 + §154

GitHub branch-protection settings are configured in the repository UI;
this doc captures the required state so it can be re-applied if reset.

## `main`

| Setting | Required value |
|---|---|
| Require pull request before merging | ✓ |
| Required approvals | 2 |
| Dismiss stale approvals on new commits | ✓ |
| Require review from CODEOWNERS | ✓ |
| Require approval of the most recent reviewable push | ✓ |
| Require conversation resolution before merging | ✓ |
| Require status checks to pass before merging | ✓ |
| Require branches to be up to date before merging | ✓ |
| Require linear history | ✓ |
| Allow force pushes | ✗ |
| Allow deletions | ✗ |

### Required status checks

These must pass before merge. Names match the workflow names in
`.github/workflows/`:

- `lint` — ruff + mypy strict
- `test-unit`
- `test-integration`
- `test-statistical` — FPR, CI coverage, recall (slow tier)
- `contract-drift-check` — RESPONSE_CONTRACT.md must change with config / schema / migrations
- `migration-directive-check` — every alembic migration carries a `deploy:` directive
- `alembic-heads-single` — `alembic heads | wc -l == 1`
- `demo-dataset-manifest` — committed CSVs match `tests/fixtures/demo_dataset_manifest.json`
- `benchmark-no-regression` — P95 phase duration not >25% over baseline (median-of-5 on self-hosted runner; only on PRs labeled `benchmark` or schedule)
- `adversarial-corpus` — every case in `tests/llm/adversarial_corpus.json` blocked by `composite_validate`
- `behavioral-corpus` — every case in `tests/llm/behavioral_corpus.json` reaches expected outcome
- `sensitive-data-egress` — `tests/integration/test_sensitive_data_egress.py` green
- `codeowners-consistency` — CODEOWNERS matches plan §121

### Per-area extra approvals (CODEOWNERS-driven)

| Path | Mandatory reviewer |
|---|---|
| `src/ive/llm/` | @security-reviewer |
| `docs/RESPONSE_CONTRACT.md` | All four area leads |
| `CODEOWNERS` | @security-reviewer |
| `alembic/` | @backend-lead AND @ml-lead |

## Hotfix flow

PRs labeled `hotfix` may be merged with **1** approval (instead of 2)
provided:

1. Full CI run is green.
2. Triggering label `hotfix` was applied by a member of the on-call
   team (not the PR author).
3. A follow-up tracking ticket is filed within 24 h.

This loosens the approval rule, **not** the status-check rule. All
gates above still apply.

## Force-push policy

- Force-push to `main`: **denied** (branch protection above).
- Force-push to feature branches: **allowed** by author until merge.
- Force-push to a release branch (`release/*`): **denied** without
  security-reviewer approval, even by the author.

## Tag protection

Tags matching `v[0-9]*` (release tags) are protected:

| Setting | Value |
|---|---|
| Allow only specific actors to create | ✓ |
| Allowed creators | @platform-lead, @ml-lead |
| Allow deletion | ✗ |

## Disaster recovery

If branch protection is accidentally cleared (admin override, repo
recreation, etc.), restore from this document. Drift from this spec
should be detected by the quarterly governance review and remediated
in the same week.

## Changelog

| Date | Change |
|---|---|
| 2026-04-28 | Initial spec. |
