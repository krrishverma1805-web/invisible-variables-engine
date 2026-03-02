# Invisible Variables Engine

> **Status:** 🚧 Active Development — v0.1.0 (Alpha)

A production-grade data science system that discovers **hidden latent variables** in datasets by analysing systematic model prediction errors. The engine runs a four-phase pipeline (Understand → Model → Detect → Construct) and surfaces explanations of _why_ a model consistently fails on certain subgroups.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Development](#development)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  Streamlit UI (8501)               │
└───────────────────────┬────────────────────────────┘
                        │ REST / WebSocket
┌───────────────────────▼────────────────────────────┐
│              FastAPI Service (8000)                │
│  ┌──────────┐  ┌────────────┐  ┌───────────────┐  │
│  │  Auth MW │  │ Rate Limit │  │ Error Handler │  │
│  └──────────┘  └────────────┘  └───────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │              API v1 Router                   │  │
│  │  /datasets  /experiments  /latent-variables  │  │
│  └──────────────────────────────────────────────┘  │
└───────────────────────┬────────────────────────────┘
                        │ Enqueue jobs
┌───────────────────────▼────────────────────────────┐
│           Redis (6379) — Broker + Cache            │
└───────────────────────┬────────────────────────────┘
                        │ Consume jobs
┌───────────────────────▼────────────────────────────┐
│              Celery Worker Pool                    │
│  ┌─────────────────────────────────────────────┐  │
│  │              IVE Engine                     │  │
│  │  Phase 1: Understand  →  Data profiling     │  │
│  │  Phase 2: Model       →  Train + residuals  │  │
│  │  Phase 3: Detect      →  Subgroup patterns  │  │
│  │  Phase 4: Construct   →  Synthesize LVs     │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────┬────────────────────────────┘
                        │ Read/Write
┌───────────────────────▼────────────────────────────┐
│         PostgreSQL (5432) — Primary DB             │
└────────────────────────────────────────────────────┘
```

### Four-Phase Pipeline

| Phase | Name           | Description                                                                          |
| ----- | -------------- | ------------------------------------------------------------------------------------ |
| 1     | **Understand** | Profile dataset; detect column types, distributions, missing values, correlations    |
| 2     | **Model**      | Train baseline models; compute cross-validated residuals                             |
| 3     | **Detect**     | Discover subgroups with high residual error via SHAP, HDBSCAN & subgroup discovery   |
| 4     | **Construct**  | Synthesise candidate latent variables; validate statistically; generate explanations |

---

## Quick Start

### Prerequisites

- Docker ≥ 24.0 and Docker Compose ≥ 2.20
- Python 3.11+ (for local development)
- Poetry (for local development)

### Using Docker Compose

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/invisible-variables-engine.git
cd invisible-variables-engine

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start all services
make dev

# 4. Run database migrations
make migrate

# 5. (Optional) Seed development data
make seed
```

Services will be available at:

| Service       | URL                        |
| ------------- | -------------------------- |
| API           | http://localhost:8000      |
| OpenAPI Docs  | http://localhost:8000/docs |
| Streamlit UI  | http://localhost:8501      |
| Celery Flower | http://localhost:5555      |

### Local Development (without Docker)

```bash
make setup          # Install dependencies and pre-commit hooks
make migrate        # Run DB migrations
make dev-local      # Start API server
make worker-local   # Start Celery worker (separate terminal)
make streamlit-local # Start Streamlit (separate terminal)
```

---

## Development

### Common Commands

```bash
make test           # Run full test suite
make test-unit      # Unit tests only
make lint           # Run ruff + mypy
make format         # Auto-format code
make makemigrations MSG="describe your change"
make migrate
make clean          # Remove caches and build artifacts
make logs           # Tail all service logs
```

### Project Structure

```
.
├── src/ive/              # Core Python package
│   ├── api/              # FastAPI routes, schemas, middleware
│   ├── core/             # Four-phase engine + orchestrator
│   ├── data/             # Ingestion, profiling, validation, preprocessing
│   ├── models/           # ML models and residual analysis
│   ├── detection/        # Subgroup discovery, clustering, SHAP
│   ├── construction/     # Latent variable synthesis + validation
│   ├── db/               # SQLAlchemy models + repositories
│   ├── workers/          # Celery app + task definitions
│   ├── storage/          # Artifact store (local / S3)
│   └── utils/            # Logging, statistics, helpers
├── streamlit_app/        # Streamlit UI
├── tests/                # Unit, integration, statistical tests
├── scripts/              # Utility scripts
├── alembic/              # Database migrations
└── docs/                 # HLD, LLD, API, User Guide
```

---

## API Reference

See [docs/API.md](docs/API.md) or the auto-generated docs at http://localhost:8000/docs.

Authentication: Pass your API key in the `X-API-Key` header.

---

## Testing

```bash
make test                 # All tests with coverage
make test-unit            # Unit tests (fast)
make test-integration     # Integration tests (needs running services)
make test-statistical     # Statistical validation tests
make test-coverage        # Generate HTML coverage report
```

Coverage target: **≥ 85%**

---

## Configuration

All configuration is environment-variable driven. See [`.env.example`](.env.example) for the full reference.

Key variables:

| Variable         | Default       | Description                        |
| ---------------- | ------------- | ---------------------------------- |
| `DATABASE_URL`   | —             | Postgres async DSN                 |
| `REDIS_URL`      | —             | Redis DSN                          |
| `SECRET_KEY`     | —             | Required in production             |
| `VALID_API_KEYS` | —             | Comma-separated API keys           |
| `ENV`            | `development` | `development\|staging\|production` |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes with tests
4. Run `make check` to verify lint + types
5. Run `make test` to verify tests pass
6. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE).
