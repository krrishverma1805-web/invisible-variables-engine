# API Reference — Invisible Variables Engine

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

All endpoints require an API key passed in the request header:

```
X-API-Key: <your-api-key>
```

Keys are configured via the `VALID_API_KEYS` environment variable (comma-separated).

**Error responses for bad auth:**

```json
{"detail": "Invalid or missing API key"}   // 401
{"detail": "Rate limit exceeded"}           // 429
```

---

## Endpoints

### Health

#### `GET /health`

Liveness probe. No auth required.

**Response 200:**

```json
{ "status": "ok", "version": "0.1.0" }
```

#### `GET /health/ready`

Readiness probe — checks DB and Redis connectivity.

**Response 200:**

```json
{ "status": "ready", "db": "ok", "redis": "ok" }
```

---

### Datasets

#### `POST /datasets`

Upload a dataset for analysis.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | CSV or Parquet file |
| `name` | string | ✅ | Human-readable name |
| `target_column` | string | ✅ | Name of the target/label column |
| `description` | string | ❌ | Optional description |

**Response 201:**

```json
{
  "id": "uuid",
  "name": "Housing Prices",
  "target_column": "price",
  "row_count": 25000,
  "column_count": 42,
  "status": "uploaded",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### `GET /datasets`

List all datasets (paginated).

**Query params:** `page=1`, `page_size=20`, `search=<str>`

#### `GET /datasets/{dataset_id}`

Get dataset details including profile summary.

#### `DELETE /datasets/{dataset_id}`

Delete a dataset and all associated experiments.

---

### Experiments

#### `POST /experiments`

Start a new IVE analysis experiment.

**Request body:**

```json
{
  "dataset_id": "uuid",
  "name": "Housing Experiment v1",
  "config": {
    "model_types": ["linear", "xgboost"],
    "cv_folds": 5,
    "random_seed": 42,
    "min_cluster_size": 10,
    "shap_sample_size": 500,
    "max_latent_variables": 5
  }
}
```

**Response 202:**

```json
{
  "id": "uuid",
  "dataset_id": "uuid",
  "status": "queued",
  "task_id": "celery-task-uuid",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### `GET /experiments`

List experiments, optionally filtered by dataset or status.

**Query params:** `dataset_id=<uuid>`, `status=running|completed|failed`

#### `GET /experiments/{experiment_id}`

Get full experiment details including phase progress.

**Response 200:**

```json
{
  "id": "uuid",
  "status": "running",
  "phase": "detect",
  "progress_pct": 65,
  "phases": {
    "understand": { "status": "completed", "duration_s": 12.3 },
    "model": { "status": "completed", "duration_s": 48.7 },
    "detect": { "status": "running", "duration_s": null },
    "construct": { "status": "pending", "duration_s": null }
  },
  "results_summary": null
}
```

#### `DELETE /experiments/{experiment_id}`

Cancel a running experiment or delete a completed one.

---

### Latent Variables

#### `GET /experiments/{experiment_id}/latent-variables`

List all discovered latent variables for an experiment.

**Response 200:**

```json
[
  {
    "id": "uuid",
    "experiment_id": "uuid",
    "rank": 1,
    "name": "Implicit Neighbourhood Quality",
    "description": "A composite signal of school ratings, green space access, and noise level that the model lacks as an explicit feature.",
    "confidence_score": 0.87,
    "effect_size": 0.34,
    "coverage_pct": 23.5,
    "candidate_features": ["zip_code", "avg_commute_mins", "nearest_park_dist"],
    "validation": {
      "bootstrap_stability": 0.91,
      "p_value": 0.0012,
      "holdout_improvement": 0.08
    }
  }
]
```

#### `GET /latent-variables/{lv_id}`

Get full latent variable details including explanation and feature importance.

#### `GET /latent-variables/{lv_id}/explanation`

Get the natural language explanation for a latent variable.

---

## WebSocket — Progress Updates

Connect to `ws://localhost:8000/ws/experiments/{experiment_id}/progress` to receive real-time progress events.

**Event format:**

```json
{
  "event": "phase_update",
  "phase": "detect",
  "progress_pct": 72,
  "message": "Running HDBSCAN clustering on residual space...",
  "timestamp": "2024-01-01T00:01:30Z"
}
```

**Event types:** `phase_update`, `phase_complete`, `experiment_complete`, `experiment_failed`

---

## Error Codes

| HTTP Status | Code                   | Description                            |
| ----------- | ---------------------- | -------------------------------------- |
| 400         | `VALIDATION_ERROR`     | Request body/params failed validation  |
| 401         | `UNAUTHORIZED`         | Missing or invalid API key             |
| 404         | `NOT_FOUND`            | Resource does not exist                |
| 409         | `CONFLICT`             | Operation conflicts with current state |
| 422         | `UNPROCESSABLE_ENTITY` | File content or schema invalid         |
| 429         | `RATE_LIMITED`         | Too many requests                      |
| 500         | `INTERNAL_ERROR`       | Server-side error                      |
| 503         | `SERVICE_UNAVAILABLE`  | Dependency (DB/Redis) unavailable      |

**Error response schema:**

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Dataset with id 'abc' not found",
    "request_id": "req_xyz"
  }
}
```
