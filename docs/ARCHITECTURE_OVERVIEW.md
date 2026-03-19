# Architecture Overview

The Invisible Variables Engine is composed of six services orchestrated through Docker Compose, implementing a four-phase ML pipeline that runs asynchronously via a Celery task queue.

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Docker Compose Network                         │
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Streamlit │    │ FastAPI  │    │  Celery   │    │  Flower  │         │
│  │   :8501   │───▶│  :8000   │───▶│  Worker   │    │  :5555   │         │
│  │    UI     │    │   API    │    │ (4 proc)  │    │ Monitor  │         │
│  └──────────┘    └────┬─────┘    └────┬──────┘    └────┬─────┘         │
│                       │               │                │               │
│                       ▼               ▼                ▼               │
│              ┌──────────────┐  ┌──────────────┐                        │
│              │  PostgreSQL  │  │    Redis     │                        │
│              │    :5432     │  │    :6379     │                        │
│              │   Database   │  │ Broker+Cache │                        │
│              └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Roles

| Service | Port | Role |
|---------|------|------|
| **FastAPI** | 8000 | REST API server. Handles dataset uploads, experiment creation, progress polling, result retrieval, and report generation. Runs Alembic migrations on startup. |
| **Celery Worker** | — | Asynchronous task executor. Runs the four-phase ML pipeline in a background process with configurable concurrency (default: 4). Listens on queues: `default`, `analysis`, `high_priority`. |
| **PostgreSQL** | 5432 | Primary data store. Holds datasets, experiments, trained model metadata, error patterns, latent variables, and residuals. SQLAlchemy 2.0 async sessions + Alembic migrations. |
| **Redis** | 6379 | Celery message broker and result backend. Also serves as a lightweight cache. Password-protected. |
| **Flower** | 5555 | Web-based Celery monitoring dashboard. Displays active tasks, worker status, and task history. Basic auth protected. |
| **Streamlit** | 8501 | Interactive UI for uploading datasets, launching experiments, and viewing results visually. Communicates with the API server. |

---

## Pipeline Phases

Every IVE experiment executes four sequential phases inside the Celery worker:

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────┐
│  Phase 1  │───▶│  Phase 2  │───▶│  Phase 3  │───▶│    Phase 4    │
│ UNDERSTAND │    │   MODEL   │    │  DETECT   │    │   CONSTRUCT   │
│            │    │           │    │           │    │ & VALIDATE    │
│ Load CSV   │    │ K-Fold CV │    │ KS Scan   │    │ Synthesize    │
│ Profile    │    │ Linear +  │    │ HDBSCAN   │    │ Bootstrap     │
│ Schema     │    │ XGBoost   │    │ Clusters  │    │ Explain       │
│ Clean      │    │ Residuals │    │ Patterns  │    │ Persist       │
└───────────┘    └───────────┘    └───────────┘    └───────────────┘
     5%               20%              55%              75% → 100%
```

### Phase 1 — Understand

Loads the CSV from artifact storage, applies schema detection, drops non-feature columns (IDs, text, datetimes), fills NaN with column medians, and splits into feature matrix `X` and target vector `y`.

### Phase 2 — Model

Trains baseline models using K-fold cross-validation (default 5 folds) with both Linear Regression and XGBoost. Collects out-of-fold predictions and computes per-sample residuals. Per-fold R² and RMSE metrics are persisted to the database.

**Why out-of-fold residuals matter:** Standard residuals (train-set) confound model overfitting with genuine patterns. Out-of-fold residuals are computed on held-out data, so any systematic structure in them reflects real model blind spots — not training noise.

### Phase 3 — Detect

Two parallel detection strategies scan the OOF residuals:

1. **Subgroup Discovery** — For every feature column, bins values into quantile groups (numeric) or per-value groups (categorical). Applies a two-sample Kolmogorov-Smirnov test to each bin's residuals vs. the global distribution, with Bonferroni correction. Retains patterns where `p < α_adj` AND `|Cohen's d| > min_effect_size`.

2. **HDBSCAN Clustering** — Applies density-based clustering to the numeric feature matrix weighted by absolute residual magnitude. Identifies geometric clusters of samples where the model systematically underperforms.

### Phase 4 — Construct & Validate

Converts detected patterns into latent variable candidates:

- **Subgroup patterns** → binary indicator variables (1 if the row matches the subgroup rule, 0 otherwise). Numeric bins store explicit interval bounds for exact reconstruction.
- **Cluster patterns** → continuous proximity scores via an inverse-distance kernel to the cluster center.

Each candidate is then validated through **bootstrap resampling**:

1. Draw `n_iterations` bootstrap samples of the original data (default: 50).
2. Re-apply the candidate's construction rule to each resample.
3. Check three survival gates: variance > floor, range > floor, support rate within bounds.
4. Compute the bootstrap presence rate (% of resamples that survived all gates).
5. Candidates above the stability threshold are marked "validated"; others are "rejected" with a diagnostic reason.

**Why bootstrap validation matters:** A pattern detected on the original sample could be an artifact of the specific data ordering, outliers, or noise alignment. Bootstrap resampling tests whether the same construction rule consistently produces a non-degenerate signal across randomised variants of the data. Patterns that only emerge in the original sample — and collapse under resampling — are rejected as unstable.

Finally, an `ExplanationGenerator` produces business-ready text for every result: polished summaries, rejection reasons, and actionable recommendations.

---

## Data Flow

```
CSV Upload ──▶ DataIngestionService ──▶ DataProfiler ──▶ PostgreSQL (Dataset)
                                                              │
                      ┌───────────────────────────────────────┘
                      ▼
              ExperimentCreate ──▶ Celery Task Dispatch
                                         │
                      ┌──────────────────┘
                      ▼
              ┌──────────────┐
              │  IVEPipeline │ (in Celery worker)
              │              │
              │  Phase 1 ──▶ Load & clean data
              │  Phase 2 ──▶ Cross-validate → OOF residuals
              │  Phase 3 ──▶ SubgroupDiscovery + HDBSCAN → patterns
              │  Phase 4 ──▶ Synthesize → Bootstrap → Explain
              │              │
              └──────┬───────┘
                     │
                     ▼
              PostgreSQL (ErrorPattern, LatentVariable, TrainedModel, Residual)
                     │
                     ▼
              API Results Endpoints ──▶ JSON / CSV / Report
                     │
                     ▼
              Streamlit UI ◀── API ──▶ Swagger Docs
```

---

## Database Schema

Core tables managed by SQLAlchemy + Alembic:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `datasets` | Uploaded CSV metadata | `name`, `target_column`, `row_count`, `schema_json`, `file_path` |
| `experiments` | Analysis run state | `dataset_id`, `status`, `progress_pct`, `config_json`, `celery_task_id` |
| `trained_models` | Per-fold model metrics | `model_type`, `fold_number`, `train_metric`, `val_metric` |
| `error_patterns` | Phase 3 detections | `pattern_type`, `effect_size`, `p_value`, `sample_count` |
| `latent_variables` | Phase 4 results | `name`, `status`, `stability_score`, `construction_rule`, `explanation_text` |
| `residuals` | Per-sample OOF errors | `model_type`, `actual_value`, `predicted_value`, `residual_value` |

---

## Authentication

The API uses a simple API key scheme via the `X-API-Key` header. Valid keys are configured in the `VALID_API_KEYS` environment variable (comma-separated). Health check endpoints are exempt from authentication.

---

## Configuration

All configuration is environment-driven via Pydantic Settings (`src/ive/config.py`). Key configuration groups:

- **Application**: `ENV`, `SECRET_KEY`, `LOG_LEVEL`, `API_PORT`
- **Database**: `DATABASE_URL`, pool size, overflow, timeout
- **Redis**: `REDIS_URL`, password
- **Celery**: concurrency, serializer, result expiry
- **ML**: default CV folds, test size, random seed, max features, HDBSCAN min cluster size, SHAP sample size
- **Storage**: `ARTIFACT_STORE_TYPE` (local/S3), `ARTIFACT_BASE_DIR`
- **Rate Limiting**: requests per window
- **Observability**: Sentry DSN, Prometheus port
