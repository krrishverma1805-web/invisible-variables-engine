# API Usage Guide

The Invisible Variables Engine exposes a RESTful JSON API under `/api/v1`. All endpoints (except health checks) require an API key.

**Base URL**: `http://localhost:8000/api/v1`
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

Include your API key in the `X-API-Key` header on every request:

```bash
curl -H "X-API-Key: dev-key-1" http://localhost:8000/api/v1/datasets/
```

Valid keys are configured via the `VALID_API_KEYS` environment variable (comma-separated). Health endpoints (`/health`, `/health/ready`) are exempt.

---

## 1. Health Checks

### Liveness

```bash
curl http://localhost:8000/api/v1/health
```

Returns `200` if the API process is running.

### Readiness

```bash
curl http://localhost:8000/api/v1/health/ready
```

Returns `200` if PostgreSQL and Redis are both reachable; `503` otherwise.

---

## 2. Datasets

### Upload a dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets/ \
  -H "X-API-Key: dev-key-1" \
  -F "file=@demo_datasets/delivery_hidden_weather.csv" \
  -F "target_column=delivery_time" \
  -F "name=Delivery Weather Demo"
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | CSV file (max 500 MB) |
| `target_column` | string | Yes | Name of the target/label column |
| `time_column` | string | No | Datetime column for temporal analysis |
| `name` | string | No | Display name (defaults to filename) |

**Response** (`201 Created`):

```json
{
  "id": "a1b2c3d4-...",
  "name": "Delivery Weather Demo",
  "target_column": "delivery_time",
  "row_count": 1000,
  "col_count": 6,
  "schema_json": { "columns": [...], "quality_score": 0.92 },
  "created_at": "2026-03-19T12:00:00Z"
}
```

### List datasets

```bash
curl -H "X-API-Key: dev-key-1" \
  "http://localhost:8000/api/v1/datasets/?skip=0&limit=20&search=delivery"
```

### Get dataset detail

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/datasets/{dataset_id}
```

### Get dataset profile

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/datasets/{dataset_id}/profile
```

Returns statistical column profiles, quality issues, recommendations, and top correlations.

### Delete a dataset

```bash
curl -X DELETE -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/datasets/{dataset_id}
```

Returns `204 No Content`. Cascade-deletes associated experiments.

---

## 3. Experiments

### Create an experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiments/ \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "a1b2c3d4-...",
    "config": {
      "analysis_mode": "demo",
      "n_folds": 5,
      "bootstrap_iterations": 50
    }
  }'
```

**Config options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `analysis_mode` | string | `"production"` | `"demo"` or `"production"` |
| `n_folds` | int | 5 | Cross-validation folds |
| `bootstrap_iterations` | int | 50 | Bootstrap resamples for validation |

**Response** (`201 Created`):

```json
{
  "id": "e5f6g7h8-...",
  "status": "queued",
  "celery_task_id": "abc123...",
  "message": "Experiment queued successfully."
}
```

### List experiments

```bash
# All experiments
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/

# Filter by dataset
curl -H "X-API-Key: dev-key-1" \
  "http://localhost:8000/api/v1/experiments/?dataset_id=a1b2c3d4-..."

# Filter by status
curl -H "X-API-Key: dev-key-1" \
  "http://localhost:8000/api/v1/experiments/?status=completed"
```

### Get experiment detail

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}
```

### Poll experiment progress

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}/progress
```

**Response:**

```json
{
  "id": "e5f6g7h8-...",
  "status": "running",
  "progress_pct": 55,
  "current_phase": "Detection"
}
```

Status values: `queued` → `running` → `completed` | `failed` | `cancelled`

### Cancel an experiment

```bash
curl -X POST -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}/cancel
```

Only works for experiments with status `queued` or `running`.

### Delete an experiment

```bash
curl -X DELETE -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}
```

---

## 4. Results

### Get error patterns

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}/patterns
```

Returns an array of detected patterns with `pattern_type`, `column_name`, `effect_size`, `p_value`, and `sample_count`.

### Get latent variables

```bash
curl -H "X-API-Key: dev-key-1" \
  "http://localhost:8000/api/v1/experiments/{experiment_id}/latent-variables?status=validated"
```

**Query parameters:**

| Field | Description |
|-------|-------------|
| `status` | Filter: `candidate`, `validated`, or `rejected` |
| `skip` | Pagination offset (default 0) |
| `limit` | Page size (default 20) |

### Get single latent variable detail

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}/latent-variables/{variable_id}
```

---

## 5. Reports

### Experiment summary (compact)

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/{experiment_id}/summary
```

Returns a JSON object with `headline`, `patterns_found`, `validated_variables`, `rejected_variables`, `summary_text`, `top_findings`, and `recommendations`.

### Full JSON report (downloadable)

```bash
curl -H "X-API-Key: dev-key-1" \
  -o report.json \
  http://localhost:8000/api/v1/experiments/{experiment_id}/report
```

Downloads a complete JSON report with experiment metadata, dataset info, all patterns, all latent variables, and the executive summary.

---

## 6. Exports

### Export patterns as CSV

```bash
curl -H "X-API-Key: dev-key-1" \
  -o patterns.csv \
  http://localhost:8000/api/v1/experiments/{experiment_id}/patterns/export
```

### Export latent variables as CSV

```bash
curl -H "X-API-Key: dev-key-1" \
  -o latent_variables.csv \
  http://localhost:8000/api/v1/experiments/{experiment_id}/latent-variables/export
```

---

## Error Responses

All error responses follow a consistent format:

```json
{
  "detail": "Dataset 'xyz' not found."
}
```

| Status Code | Meaning |
|-------------|---------|
| `400` | Bad request / validation error |
| `401` | Missing or invalid API key |
| `404` | Resource not found |
| `409` | Conflict (e.g., cancelling a completed experiment) |
| `413` | File too large (> 500 MB) |
| `415` | Unsupported file type (only .csv accepted) |
| `422` | Unprocessable entity (validation failure) |
| `500` | Internal server error |

---

## Pagination

All list endpoints support pagination via query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skip` | `0` | Number of records to skip |
| `limit` | `20` | Maximum records to return |

Paginated responses include `total`, `skip`, and `limit` fields alongside the data array.
