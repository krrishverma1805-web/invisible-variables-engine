# Runbook — Stability Calibration Grid Mismatch

**Trigger:** structlog event `stability_calibration_grid_mismatch` (no
Prometheus alert by default — a quiet correctness signal).
**Severity:** warning
**Owner:** ml-platform

## Symptom

An experiment's active config (e.g. `bootstrap_iterations`,
`min_subgroup_size`, `min_effect_size`, `n_rows` bucket, `mode`,
`problem_type`) does not match any entry in
`data/calibration/stability_thresholds.json`. The system has fallen
through to one of:

* `"adaptive"` mode — runs online permutation calibration on this
  experiment (5–10 min added runtime, but FPR guarantee is per-experiment
  exact).
* `"fixed"` mode — uses `default_stability_threshold` (no FPR
  guarantee).

## Likely cause

1. **New config knob** — someone added a tuning axis (e.g. a new
   `bootstrap_iterations` default) that wasn't in the calibrated grid.
2. **User-supplied config off-grid** — an experiment with
   `min_subgroup_size=15` when the grid only covers 30/50/100.
3. **Calibration JSON outdated** — schema bumped (e.g. v1 → v2 added
   classification axis); stale JSON in the deployment.
4. **Schema-version skew** — `config_grid.schema_version` doesn't
   match what `load_calibration_table()` expects.

## Diagnosis

```bash
# What does the JSON think it covers?
jq '.config_grid' data/calibration/stability_thresholds.json

# What grid did the experiment ask for?
psql "$DATABASE_URL" -c \
  "SELECT id, config FROM experiments WHERE id = '<exp_id>';"

# How many adaptive-fallback events in last 24h?
psql "$DATABASE_URL" -c \
  "SELECT count(*) FROM experiment_events
   WHERE event_type='stability_calibration_grid_mismatch'
     AND created_at > now() - interval '24 hours';"
```

## Mitigation

### Option 1: Regenerate the calibration grid (preferred)

```bash
# Regenerate against current default config
make calibrate-stability
# ~30 min single-threaded; CI does NOT run this.

# Inspect diff
git diff data/calibration/stability_thresholds.json
```

Commit + tag release notes with `[calibration-update]`.

### Option 2: Force-route to adaptive

```bash
# Temporarily set per-deployment
STABILITY_CALIBRATION_STRATEGY=adaptive
# pays runtime cost (~5–10 min/experiment) for exact FPR guarantee
```

### Option 3: Force-route to fixed (LAST RESORT)

```bash
STABILITY_CALIBRATION_STRATEGY=fixed
# loses FPR guarantee — only acceptable when adaptive runtime is
# unacceptable AND the operator has decided this trade-off explicitly.
```

## Postmortem checklist

- [ ] Was the new config dimension added to
      `scripts/calibrate_stability_thresholds.py` `--*` flags?
- [ ] Is `config_grid.schema_version` bumped on every grid change?
- [ ] Did CI fail-fast on schema-version mismatch (`load_calibration_table()`
      raising on `v1` JSON when code expects `v2`)?
- [ ] Was the release tagged with `[calibration-update]`?
