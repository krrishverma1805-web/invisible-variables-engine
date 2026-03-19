# Invisible Variables Engine

**Discover what your model can't see.**

The Invisible Variables Engine (IVE) is a production-grade machine learning platform that automatically discovers hidden latent variables in tabular datasets. It works by training baseline models, analyzing systematic patterns in prediction errors, and constructing new variables that capture unmeasured conditions — factors your data does not explicitly record but that measurably influence outcomes.

---

## Key Features

- **Automated Latent Variable Discovery** — upload a CSV, specify a target, and IVE handles the rest: profiling, modeling, residual analysis, pattern detection, variable synthesis, and statistical validation.
- **Out-of-Fold Residual Analysis** — K-fold cross-validation ensures residual patterns reflect genuine model blind spots, not training artifacts.
- **Subgroup & Cluster Detection** — Bonferroni-corrected KS tests identify feature-value subgroups with anomalous errors; HDBSCAN finds geometric clusters of high-error samples.
- **Bootstrap Stability Validation** — every candidate latent variable is stress-tested across 50+ bootstrap resamples with triple-gate survival checks (variance, range, support).
- **Business-Ready Explanations** — every finding is translated into clear, non-technical language suitable for executives, analysts, and domain experts.
- **Full REST API** — 17 endpoints covering datasets, experiments, progress monitoring, results, reports, and CSV exports.
- **Real-Time Monitoring** — Celery Flower dashboard for task monitoring; progress polling for live experiment tracking.
- **Interactive Streamlit UI** — visual interface for uploading data, launching experiments, and exploring results.

---

## Architecture Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI 0.111 · Pydantic v2 · Uvicorn |
| **Task Queue** | Celery 5.4 · Redis 7 |
| **Database** | PostgreSQL 16 · SQLAlchemy 2.0 (async) · Alembic |
| **ML** | scikit-learn 1.5 · XGBoost 2.0 · HDBSCAN · SciPy · SHAP |
| **UI** | Streamlit 1.35 · Plotly |
| **Observability** | structlog · Sentry (optional) · Celery Flower |
| **Infrastructure** | Docker · Docker Compose |

---

## Folder Structure

```
invisible-variables-engine/
├── src/ive/                   # Core Python package
│   ├── api/v1/                #   FastAPI routers, schemas, dependencies
│   ├── config.py              #   Pydantic Settings (env-based)
│   ├── construction/          #   Phase 4: synthesis, bootstrap, explanation
│   ├── core/                  #   Pipeline orchestrator
│   ├── data/                  #   Phase 1: ingestion, profiling, preprocessing
│   ├── db/                    #   SQLAlchemy models, repositories, Alembic
│   ├── detection/             #   Phase 3: subgroup discovery, HDBSCAN clustering
│   ├── main.py                #   FastAPI app factory
│   ├── models/                #   Phase 2: Linear, XGBoost, cross-validator
│   ├── storage/               #   Artifact store (local / S3)
│   ├── utils/                 #   Logging, reporting helpers
│   └── workers/               #   Celery app, task definitions
├── streamlit_app/             # Streamlit UI
├── demo_datasets/             # 5 synthetic datasets with ground-truth metadata
├── tests/                     # Test suite (unit, integration, statistical)
├── alembic/                   # Database migration scripts
├── scripts/                   # DB seeding, data generation utilities
├── docs/                      # Extended documentation
├── docker-compose.yml         # Multi-service orchestration
├── Dockerfile                 # Multi-stage build (api, worker, streamlit)
├── Makefile                   # Developer workflow commands
└── pyproject.toml             # Poetry project + tool configuration
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2+
- 4 GB RAM minimum (8 GB recommended)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/invisible-variables-engine.git
cd invisible-variables-engine
cp .env.example .env
```

### 2. Launch all services

```bash
docker compose up --build -d
```

### 3. Verify

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Readiness (DB + Redis)
curl http://localhost:8000/api/v1/health/ready
```

### 4. Access

| Service | URL |
|---------|-----|
| **API** | [http://localhost:8000](http://localhost:8000) |
| **API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Streamlit UI** | [http://localhost:8501](http://localhost:8501) |
| **Flower Dashboard** | [http://localhost:5555](http://localhost:5555) |

### 5. Run a demo experiment

```bash
# Upload a demo dataset
curl -X POST http://localhost:8000/api/v1/datasets/ \
  -H "X-API-Key: dev-key-1" \
  -F "file=@demo_datasets/delivery_hidden_weather.csv" \
  -F "target_column=delivery_time"

# Create experiment (use the dataset_id from the response)
curl -X POST http://localhost:8000/api/v1/experiments/ \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "<DATASET_ID>", "config": {"analysis_mode": "demo"}}'
```

---

## Docker Commands

```bash
make dev              # Build and start all services
make down             # Stop all services
make restart          # Restart all services
make logs             # Tail logs from all services
make logs-api         # Tail API logs only
make logs-worker      # Tail worker logs only
make down-volumes     # Stop and remove all data (destructive)
make ps               # Show running service status
```

---

## Screenshots

> **Streamlit Dashboard** — upload datasets, launch experiments, and explore discovered latent variables in an interactive visual interface.

> **API Swagger UI** — full OpenAPI documentation available at `/docs` with "Try it out" for every endpoint.

> **Flower Dashboard** — real-time Celery task monitoring at `localhost:5555`.

---

## Testing

The project includes a comprehensive test suite organized into three tiers:

```bash
make test             # Run all 201 tests with coverage
make test-unit        # Unit tests only (~150 tests, fast)
make test-integration # Integration tests (requires running services)
make test-statistical # Statistical validation tests
make test-fast        # All tests except slow statistical tests
make test-coverage    # Generate HTML coverage report
```

| Tier | Count | Description |
|------|-------|-------------|
| **Unit** | ~150 | In-process, no DB/Redis. Covers models, detection, construction, profiling. |
| **Integration** | ~30 | Requires Docker services. Tests API endpoints, job processing, E2E pipeline. |
| **Statistical** | ~20 | Numerical accuracy: reproducibility, false-positive control, ground-truth recovery. |

---

## Demo Datasets

Five synthetic datasets are included in `demo_datasets/`, each containing a known hidden variable for validation:

| Dataset | Target | Hidden Variable | Affected % | Detection Type |
|---------|--------|----------------|------------|----------------|
| `delivery_hidden_weather` | `delivery_time` | Storm delay zone (distance > 10 & traffic > 7.5) | 19.5% | Subgroup |
| `healthcare_hidden_risk` | `recovery_days` | Post-surgery complication (BMI > 30 & BP > 150) | 22.3% | Subgroup |
| `manufacturing_hidden_shift` | `defect_rate` | Night shift instability (vibration > 7 & humidity > 75) | 10.7% | Subgroup |
| `retail_hidden_promo` | `spend_amount` | Premium promo eligibility (loyalty > 0.8 & basket > 8) | 8.7% | Subgroup |
| `no_hidden_random_noise` | `target` | None (control dataset) | 0% | — |

---

## Demo vs Production Mode

IVE operates in two analysis modes, controlled by the `analysis_mode` field in the experiment configuration:

| Aspect | Demo Mode | Production Mode |
|--------|-----------|-----------------|
| **Purpose** | Synthetic datasets, demos, exploration | Real-world data with rigorous gating |
| **Bootstrap stability threshold** | 0.50 (50%) | 0.70 (70%) |
| **Minimum variance floor** | 1e-7 | 1e-5 |
| **Minimum score range** | 0.01 | 0.05 |
| **Support rate window** | 0.5%–98% | 1%–95% |
| **Subgroup min samples** | 20 | 30 |
| **Effect size threshold** | 0.15 | 0.20 |

Use **demo mode** for demonstrations and synthetic datasets. Use **production mode** for real-world analyses where false positives carry operational cost.

---

## Future Roadmap

- **Multi-target analysis** — run IVE on multiple targets simultaneously to find shared latent structure.
- **Time-series latent variables** — temporal drift detection and change-point analysis on residuals.
- **Automated re-training** — close the loop by feeding discovered variables back into model training.
- **Interactive SHAP explorer** — per-latent-variable feature importance visualization in Streamlit.
- **S3 artifact store** — production storage backend for model artifacts and datasets.
- **Kubernetes deployment** — Helm chart and horizontal pod autoscaling for Celery workers.
- **Webhook notifications** — notify external systems when experiments complete.
- **Multi-tenant isolation** — per-organization dataset and experiment scoping.

---

## License

MIT — see [LICENSE](LICENSE) for details.
