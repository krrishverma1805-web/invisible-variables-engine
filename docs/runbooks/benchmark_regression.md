# Runbook — Benchmark Regression

**Alert:** `ive_phase_p95_duration_regression` (Prometheus) **or**
GitHub Actions `benchmark` workflow fails with >25% delta.
**Severity:** warning
**Owner:** ml-platform

## Symptom

Phase P95 wall-clock duration exceeds the SLO (plan §C3 Performance SLOs)
or the benchmark CI run on a PR/scheduled job exceeds the recorded
baseline by >25%.

## Likely cause (rank-ordered)

1. **Algorithmic regression** — a new feature added work to a hot path
   (e.g. K=5 cross-fitting now runs SHAP 5× instead of once).
2. **Dependency upgrade** — XGBoost / scikit-learn / pandas minor bump
   regressing performance.
3. **Configuration drift** — `hpo_n_trials` or `bootstrap_iterations`
   accidentally raised in a non-perf-critical config.
4. **Runner contention** — non-self-hosted runner inherited noisy-
   neighbor CPU. Self-hosted runner setup (plan §151) eliminates this;
   if the alert fires there, treat as real regression.

## Diagnosis

```bash
# Compare baseline vs latest
diff -u benchmark_results/baseline.json benchmark_results/latest.json

# Per-phase split (which phase regressed?)
jq '.phases | to_entries | map({phase: .key, p95_seconds: .value.p95})' \
  benchmark_results/baseline.json
jq '.phases | to_entries | map({phase: .key, p95_seconds: .value.p95})' \
  benchmark_results/latest.json

# What landed since baseline?
git log --oneline benchmark_results/baseline.json..HEAD -- src/ | head -30
```

## Mitigation

### If regression is small (25–50%)

- Bisect to the offending PR.
- Profile the hot path:
  ```bash
  PYTHONPATH=src poetry run python -m cProfile -o /tmp/p.prof \
    -m ive.benchmark.run_one demo_datasets/delivery_hidden_weather.csv
  PYTHONPATH=src poetry run snakeviz /tmp/p.prof
  ```
- Decide: optimise, opt-in flag, or accept and update baseline.

### If regression is intentional (e.g. K=5 cross-fitting in B6)

```bash
make benchmark            # writes benchmark_results/latest.json
make update-benchmark-baseline
# Manual review + commit. Tag PR with [benchmark-baseline-update].
git diff benchmark_results/baseline.json
git commit -m "chore(benchmark): update baseline after K=5 cross-fitting

Cross-fit scoring 5× the modeling cost; SLO updated in
docs/RESPONSE_CONTRACT.md §C3 to match."
```

Update the SLO table in `docs/RESPONSE_CONTRACT.md` (plan §173) so
ops + integrators see the new ceiling.

### If false alarm (CI runner noise)

```bash
# Re-run the workflow once. If it passes, the noise was transient.
# If it fails again, treat as real.
```

## Postmortem checklist

- [ ] Was the regression behind a feature flag (so production stayed
      fast on the old default)?
- [ ] Did `benchmark_results/baseline.json` get updated in the same PR
      as the cause?
- [ ] If the change increased SLO ceilings, was `docs/RESPONSE_CONTRACT.md`
      §C3 updated in the same PR?
- [ ] Add a microbenchmark to `tests/statistical/` if the regression
      came from a hot path that lacked one.
