# Runbook — FPR Sentinel Red

**Alert:** `ive_fpr_sentinel_red`
**Severity:** critical
**Owner:** ml-platform

## Symptom

Gauge `ive_fpr_sentinel_value > 0.07` for ≥1 hour. The nightly job
(`ive.workers.tasks.fpr_sentinel_run`, 02:30 UTC) measured an empirical
false-positive rate above 7% on synthetic noise — the bootstrap-validator
threshold is over-firing.

## Likely cause (rank-ordered)

1. **Calibration regression** — `data/calibration/stability_thresholds.json`
   was edited (or schema-version bumped) without re-running calibration
   against the current default config.
2. **Detection-config drift** — someone shipped a change to
   `subgroup_discovery.py`, `bootstrap_validator.py`, or the BH-correction
   family scope (B7/B8) without re-running calibration.
3. **Sentinel itself drifted** — the sentinel uses a stripped-down KS
   subgroup scan as a proxy. If the proxy now over-fires while the real
   pipeline is still well-calibrated, we'd false-alarm.
4. **Statistical noise** — single-run upper bound at 7% is reasonable
   guard; the rolling 14-day window (plan §190) suppresses one-off
   spikes. If this fires inside that window, it's probably real.

## Diagnosis

```bash
# Recent sentinel results (Prometheus)
sum by (status) (rate(ive_fpr_sentinel_runs_total[7d]))

# Manual replay
PYTHONPATH=src poetry run python -c \
  "from ive.observability.fpr_sentinel import run_sentinel; \
   import json; \
   r = run_sentinel(n_seeds=20); \
   print(json.dumps({'fpr': r.empirical_fpr, 'upper95': r.upper_95_ci, \
                     'fp_runs': r.n_false_positive_runs, \
                     'status': r.status}, indent=2))"

# What changed in detection logic recently?
git log --oneline --since="2 weeks ago" -- \
  src/ive/detection/ src/ive/construction/bootstrap_validator.py \
  src/ive/construction/stability_calibration.py \
  data/calibration/

# Re-run calibration against current default config (~30 min)
make calibrate-stability
git diff data/calibration/stability_thresholds.json
```

## Mitigation

### Option 1 — Calibration drift (most common)

```bash
make calibrate-stability
# Inspect diff; if the new thresholds are tighter, commit + tag
# release notes with [calibration-update].
```

### Option 2 — Detection-logic regression

```bash
# Bisect the detection-module changes since last green sentinel
git bisect start HEAD <last-green-commit>
git bisect run python -c \
  "from ive.observability.fpr_sentinel import run_sentinel; \
   r = run_sentinel(n_seeds=20); \
   import sys; sys.exit(0 if r.passed else 1)"
```

Revert the offending PR; tighten the test that should have caught it.

### Option 3 — Quarantine

```bash
# Disable the FPR sentinel beat task while you investigate.
# Edit docker-compose.yml to drop the celery-beat service, OR
# remove the schedule entry from celery_app.conf.beat_schedule.
```

⚠️ This silences the watchdog. Time-box the quarantine; if it lasts
>48 h, escalate.

## Postmortem checklist

- [ ] Did the rolling 14-day window (plan §190) trigger before single-night?
- [ ] Was the offending change behind a feature flag?
- [ ] Update `tests/statistical/test_stability_fpr.py` with a regression
      case for the cause.
- [ ] If detection-logic change: was a calibration re-run mandated by
      the CI gate? If not, add it.
