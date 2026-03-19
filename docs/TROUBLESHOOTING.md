# Troubleshooting

This guide covers common issues when running the Invisible Variables Engine and how to resolve them.

---

## Container Issues

### Containers keep restarting

**Symptom:** `docker compose ps` shows services in a restart loop.

**Causes and fixes:**

1. **Missing `.env` file** — Docker Compose reads variables from `.env`. Ensure it exists:
   ```bash
   cp .env.example .env
   ```

2. **Port conflict** — another process occupies 8000, 5432, 6379, 5555, or 8501:
   ```bash
   lsof -i :8000  # check which process holds the port
   ```
   Either stop the conflicting process or change the port in `.env` (e.g., `API_PORT=8001`).

3. **Database migration failure** — the API container runs `alembic upgrade head` on startup. If the migration fails, check:
   ```bash
   docker compose logs api | grep -i "alembic\|migration\|error"
   ```
   Fix: run migrations manually after the database is stable:
   ```bash
   docker compose exec api alembic upgrade head
   ```

4. **Insufficient memory** — ML workloads (XGBoost, HDBSCAN, SHAP) require memory. If the worker is killed by OOM:
   ```bash
   docker compose logs worker | tail -20
   ```
   Increase Docker's memory allocation to at least 4 GB (8 GB recommended).

---

## Redis Auth Issues

### `NOAUTH Authentication required`

**Symptom:** Worker or API logs show Redis authentication errors.

**Cause:** The `REDIS_URL` does not include the password, or the password doesn't match `redis-server --requirepass`.

**Fix:** Ensure `REDIS_URL` in `.env` includes the password:
```bash
REDIS_URL=redis://:redispassword@localhost:6379/0
```

Inside Docker Compose, the URL should reference the `redis` service:
```bash
REDIS_URL=redis://:redispassword@redis:6379/0
```

### `Connection refused` to Redis

**Symptom:** API readiness check or Celery worker fails to connect to Redis.

**Fix:** Verify Redis is running and healthy:
```bash
docker compose ps redis
docker compose exec redis redis-cli -a redispassword ping
```

---

## PostgreSQL Auth Issues

### `password authentication failed for user "ive"`

**Symptom:** API or Alembic fails to connect to the database.

**Cause:** Password mismatch between `.env` and the initialized database.

**Fix:** If this is a fresh setup, remove the Postgres volume and rebuild:
```bash
docker compose down -v
docker compose up --build -d
```

If the database has existing data you need to preserve, update the password inside the container:
```bash
docker compose exec postgres psql -U ive -d ive_db
# Inside psql:
ALTER USER ive WITH PASSWORD 'new-password';
```
Then update `POSTGRES_PASSWORD` in `.env` to match.

### `database "ive_db" does not exist`

**Fix:** The database is created automatically when the Postgres container starts with the `POSTGRES_DB` environment variable. Remove and recreate the volume:
```bash
docker compose down -v
docker compose up --build -d
```

---

## File Mount Issues

### `FileNotFoundError` for demo datasets

**Symptom:** Worker logs show file-not-found errors when loading datasets.

**Cause:** The `demo_datasets/` directory is not mounted into the container.

**Fix:** Verify the volume mount in `docker-compose.yml`:
```yaml
volumes:
  - ./demo_datasets:/app/demo_datasets
```

Then restart:
```bash
docker compose restart worker
```

### Dataset upload succeeds but experiment fails with "file not found"

**Cause:** The `artifact_data` volume is not shared between the `api` and `worker` containers.

**Fix:** Both `api` and `worker` must mount the same `artifact_data` volume:
```yaml
# api
volumes:
  - artifact_data:/app/artifacts

# worker
volumes:
  - artifact_data:/app/artifacts
```

---

## Permissions on `/app/artifacts`

### `PermissionError: [Errno 13] Permission denied: '/app/artifacts/...'`

**Cause:** The container user does not have write access to the artifacts directory.

**Fix:**

1. Check ownership inside the container:
   ```bash
   docker compose exec api ls -la /app/artifacts
   ```

2. Fix permissions:
   ```bash
   docker compose exec api chown -R nobody:nogroup /app/artifacts
   docker compose exec api chmod -R 755 /app/artifacts
   ```

3. If using a named volume, permissions are typically correct by default. If using a bind mount, ensure the host directory is writable by the container user.

---

## No Patterns Found

**Symptom:** Experiment completes but reports 0 detected patterns.

**Possible causes:**

1. **The model fits well** — if the baseline model explains most variance, residuals are random noise with no systematic structure. This is actually a valid finding; the `no_hidden_random_noise.csv` demo validates this behavior.

2. **Effect size too small** — the minimum Cohen's d for detection is 0.20 in production mode and 0.15 in demo mode. If the hidden signal is weak, it may fall below the threshold.

3. **Too few samples** — subgroup discovery requires a minimum number of samples per bin (default 30 in production, 20 in demo). Small datasets may not have enough rows in each quantile bin.

4. **Wrong target column** — verify the `target_column` matches the actual column name in the CSV (case-sensitive).

**Calibration guidance:**
- For demo datasets, always use `analysis_mode: "demo"`.
- For real-world data, if too few patterns emerge, consider:
  - Reducing `n_folds` from 5 to 3 (more residual samples per fold)
  - Increasing the dataset size if possible
  - Checking data quality (excessive NaN columns, low-variance features)

---

## No Validated Latent Variables

**Symptom:** Patterns are detected, but all candidates are rejected during bootstrap validation.

**Possible causes:**

1. **Patterns are noise** — the detected pattern exists in the original sample but does not survive bootstrap resampling. This is the system working as intended; it correctly identifies unstable signals.

2. **Subgroup too small** — if the affected subgroup has very few rows (< 1% of the dataset in production mode), it may fail the `min_support_rate` gate.

3. **Subgroup too broad** — if the subgroup captures > 95% of rows (production) or > 98% (demo), it fails the `max_support_rate` gate.

4. **Wrong analysis mode** — using production mode on demo datasets may reject candidates that would pass in demo mode.

**Diagnostic steps:**

1. Check the `rejection_reason` field on rejected candidates:
   ```bash
   curl -H "X-API-Key: dev-key-1" \
     "http://localhost:8000/api/v1/experiments/<ID>/latent-variables?status=rejected" \
     | python3 -m json.tool
   ```

2. Look for `bootstrap_diagnostics` in the latent variable detail, which includes:
   - `preflight_support` — support on original data (should be > 0)
   - `mean_bootstrap_variance` — average variance across resamples
   - `mean_bootstrap_support` — average support across resamples
   - `fail_variance`, `fail_range`, `fail_support_low`, `fail_support_high` — per-gate failure counts

3. If `preflight_support` is 0, the construction rule itself is failing to reconstruct. Check the worker logs for warnings prefixed with `ive.bootstrap.preflight_zero`.

---

## Re-Running Demo Datasets

To re-run a demo from scratch:

### Option 1 — Delete and re-upload

```bash
# Delete the existing experiment
curl -X DELETE -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/<EXPERIMENT_ID>

# Delete the dataset
curl -X DELETE -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/datasets/<DATASET_ID>

# Upload fresh
curl -X POST http://localhost:8000/api/v1/datasets/ \
  -H "X-API-Key: dev-key-1" \
  -F "file=@demo_datasets/delivery_hidden_weather.csv" \
  -F "target_column=delivery_time"
```

### Option 2 — Full reset

```bash
# Remove all containers and volumes, then rebuild
make down-volumes
make dev
```

This destroys all stored data and starts fresh.

### Option 3 — Re-run experiment on existing dataset

Simply create a new experiment for the same dataset:

```bash
curl -X POST http://localhost:8000/api/v1/experiments/ \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "<EXISTING_DATASET_ID>", "config": {"analysis_mode": "demo"}}'
```

---

## Calibration Guidance

If IVE consistently produces unexpected results on your dataset, consider adjusting these parameters in the experiment config:

| Parameter | Effect of Increasing | Effect of Decreasing |
|-----------|---------------------|---------------------|
| `n_folds` | More robust residuals, fewer residual samples per fold | More residual samples, slightly less robust |
| `bootstrap_iterations` | More stable presence rate estimates, slower | Faster, noisier estimates |
| `analysis_mode: "demo"` | n/a | Relaxes all thresholds for synthetic data |

For real-world datasets:

- **Small datasets (< 500 rows)**: Use `demo` mode, reduce `n_folds` to 3.
- **Medium datasets (500–5000 rows)**: Use `production` mode with default settings.
- **Large datasets (> 5000 rows)**: Use `production` mode. Consider increasing `bootstrap_iterations` to 100 for tighter confidence.

---

## Getting Help

1. **Check logs**: `make logs` or `docker compose logs <service>`
2. **API docs**: Visit `http://localhost:8000/docs` for interactive endpoint testing
3. **Worker debug**: Set `LOG_LEVEL=DEBUG` in `.env` and restart the worker for verbose pipeline tracing
4. **Database state**: Connect directly via `docker compose exec postgres psql -U ive -d ive_db`
