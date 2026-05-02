# AGENTS.md — Invisible Variables Engine

> A single, authoritative briefing for any human or AI agent working in this repository.
> Read this once and you should be able to navigate, run, extend, and reason about every part of the system without further onboarding.

This document follows the [AGENTS.md](https://agentsmd.org) convention: a top-level operating manual that any coding agent (Claude Code, Cursor, Copilot, Gemini, etc.) can consume to act competently in this codebase. It is written to be exhaustive but navigable — read top-to-bottom for the first orientation, then jump by section.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Mental Model — What IVE Actually Does](#2-the-mental-model)
3. [Quick Start (Sixty Seconds)](#3-quick-start)
4. [System Architecture](#4-system-architecture)
5. [The Five-Phase Pipeline (deep dive)](#5-the-five-phase-pipeline)
6. [Repository Layout (every directory)](#6-repository-layout)
7. [Source Code Reference (every file)](#7-source-code-reference)
8. [REST API Reference (every endpoint)](#8-rest-api-reference)
9. [Database Schema](#9-database-schema)
10. [Background Jobs — Celery](#10-background-jobs)
11. [Configuration & Environment](#11-configuration)
12. [Streamlit UI](#12-streamlit-ui)
13. [Test Suite](#13-test-suite)
14. [Demo Datasets](#14-demo-datasets)
15. [Operations — Docker, CI, Migrations](#15-operations)
16. [Documentation Index (`docs/`)](#16-documentation-index)
17. [Coding Conventions & House Rules](#17-coding-conventions)
18. [Glossary](#18-glossary)
19. [Troubleshooting Pointers](#19-troubleshooting)
20. [How to Extend IVE](#20-how-to-extend-ive)

---

## 1. Executive Summary

**Project:** Invisible Variables Engine (IVE)
**Tagline:** *Discover what your model can't see.*
**Version:** `0.1.0` (Alpha)
**License:** MIT
**Language:** Python 3.11
**Build system:** Poetry (PEP 517/518)

IVE is a production-grade machine-learning platform that **automatically discovers hidden latent variables in tabular datasets**. It does this by training baseline models, analyzing systematic patterns in their out-of-fold prediction errors, and constructing new variables that capture unmeasured conditions — factors a dataset does not explicitly record but that measurably influence outcomes.

A typical question IVE answers:

> *"My delivery-time model has a 12-minute MAE. Where is that error concentrated, and is there a hidden driver — like a storm zone, a problem shift, a bad SKU class — that we should be measuring?"*

It runs as four cooperating long-lived services (API, Worker, Postgres, Redis) plus an optional Streamlit dashboard, packaged via Docker Compose for one-command spin-up.

| Layer            | Technology                                                       |
| ---------------- | ---------------------------------------------------------------- |
| API              | FastAPI 0.111 · Pydantic v2 · Uvicorn                            |
| Task Queue       | Celery 5.4 · Redis 7 · Flower                                    |
| Database         | PostgreSQL 16 · SQLAlchemy 2.0 (async) · asyncpg · Alembic       |
| ML / Statistics  | scikit-learn 1.5 · XGBoost 2.0 · SHAP · HDBSCAN · SciPy · statsmodels |
| Data             | pandas 2.2 · polars 0.20 · numpy 1.26                            |
| UI               | Streamlit 1.35 · Plotly · Altair                                 |
| Observability    | structlog · Sentry (optional) · Celery Flower                    |
| Infra            | Docker · Docker Compose · GitHub Actions                         |

---

## 2. The Mental Model

IVE's central insight: **a model's residuals (its mistakes) are a free source of signal about variables you didn't measure**. If your model is systematically wrong for a *particular slice* of your data, that slice is being driven by something the model can't see.

The pipeline operationalizes this in five strict stages:

```
   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ 1. UNDER-  │   │ 2. MODEL   │   │ 3. DETECT  │   │ 4. CONSTRUCT│  │ 5. EVALUATE│
   │    STAND   │──▶│ (CV +      │──▶│ (residual  │──▶│ (synthesize │─▶│ (holdout    │
   │ (ingest +  │   │  residuals)│   │  patterns) │   │  + bootstrap│  │  uplift,    │
   │  profile)  │   │            │   │            │   │  validate)  │  │  optional)  │
   └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
```

Every candidate latent variable goes through three independent gates before it is persisted as **validated**:

1. **Statistical gate** — Cohen's *d* effect size, Bonferroni / Benjamini-Hochberg corrected p-values.
2. **Stability gate** — bootstrap presence rate over 50+ resamples (≥0.70 production / ≥0.50 demo).
3. **Plausibility gate** — causal heuristics (no reverse causality, low partial correlation with existing features).

Variables that fail any gate are not silently dropped: they are persisted with a `status="rejected"` and a hedged, business-readable explanation of why.

---

## 3. Quick Start

### Prerequisites
- Docker Desktop / Docker Engine + Docker Compose v2+
- 8 GB RAM recommended (4 GB minimum)

### One-command bring-up
```bash
git clone <repo-url> invisible-variables-engine
cd invisible-variables-engine
cp .env.example .env
make dev          # docker compose up --build -d
```

### Service endpoints
| Service        | URL                                  | Notes                          |
| -------------- | ------------------------------------ | ------------------------------ |
| API            | http://localhost:8000                | FastAPI, hot-reload in dev     |
| Swagger Docs   | http://localhost:8000/docs           | "Try it out" for every route   |
| Streamlit      | http://localhost:8501                | Visual workbench               |
| Flower         | http://localhost:5555                | Celery task monitor            |
| PostgreSQL     | localhost:5432                       | `ive` / `ivepassword` (dev)    |
| Redis          | localhost:6379                       | password-protected             |

### Smoke-test an experiment
```bash
# 1. Upload a demo dataset
curl -X POST http://localhost:8000/api/v1/datasets/ \
  -H "X-API-Key: dev-key-1" \
  -F "file=@demo_datasets/delivery_hidden_weather.csv" \
  -F "target_column=delivery_time"

# 2. Create an experiment (replace <DATASET_ID>)
curl -X POST http://localhost:8000/api/v1/experiments/ \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"<DATASET_ID>","config":{"analysis_mode":"demo"}}'

# 3. Watch progress
curl -H "X-API-Key: dev-key-1" \
     http://localhost:8000/api/v1/experiments/<EXPERIMENT_ID>/progress
```

### Most useful Make targets
```bash
make help              # list all commands
make dev               # bring up the full stack
make logs              # tail all logs
make logs-api          # tail API only
make logs-worker       # tail worker only
make test              # full test suite with coverage
make test-unit         # fast unit tests (~150)
make test-integration  # requires services running
make lint              # ruff + mypy
make format            # ruff format + autofix
make migrate           # alembic upgrade head
make seed              # populate dev DB
make down              # stop everything
make down-volumes      # ⚠️ destructive — wipes DB volumes
```

---

## 4. System Architecture

### Service topology (logical)
```
                           ┌────────────────────────────┐
                           │       Streamlit UI         │
                           │   (5 pages, components/)   │
                           └────────────┬───────────────┘
                                        │ HTTP (X-API-Key)
                                        ▼
┌──────────────────┐  enqueue   ┌────────────────────────────┐  AsyncPG  ┌──────────────┐
│  Celery Worker   │◀───────────│         FastAPI            │──────────▶│ PostgreSQL   │
│  (analysis,      │            │  /api/v1/{datasets,        │           │ (datasets,   │
│   default,       │  pubsub    │   experiments,             │           │  experiments,│
│   high_priority) │◀──────────▶│   latent-variables,        │           │  patterns,   │
│                  │            │   health}                  │           │  LVs, events)│
└────────┬─────────┘            └────────────┬───────────────┘           └──────────────┘
         │                                   │
         │ broker + result backend           │ rate-limit / cache
         └───────────┬───────────────────────┘
                     ▼
                ┌────────────┐
                │   Redis    │
                │  db0 broker│
                │  db1 result│
                └────────────┘
                     ▲
                     │ broker URL
                ┌────┴───────┐
                │  Flower    │
                └────────────┘

artifact_data volume ◀── shared between API + Worker (CSV, model pickles)
```

### Process model
- **API** is async (asyncpg, async SQLAlchemy). It never runs ML code in-request — every long-running operation is enqueued as a Celery task.
- **Worker** is sync (Celery default). When it needs async DB access it bridges via `asyncio.run()` inside the task body.
- **Auth** is API-key based via the `X-API-Key` header (set in `valid_api_keys`); health and docs paths are exempted.
- **Rate limit** is sliding-window per IP (`rate_limit_requests` per `rate_limit_window` seconds).

### Storage model
- **Datasets** (raw CSVs) are persisted to `artifact_data` volume by `LocalArtifactStore` (`src/ive/storage/artifact_store.py`). Optional S3 backend via `ARTIFACT_STORE_TYPE=s3`.
- **Trained models** and **OOF residuals** are persisted into Postgres (not pickle files) for queryability.
- **Result backend** (Celery) is Redis db `/1`; broker is Redis db `/0`.

---

## 5. The Five-Phase Pipeline

Implemented in [`src/ive/core/pipeline.py`](src/ive/core/pipeline.py) by the `IVEPipeline` class. The orchestrator emits an `ExperimentEvent` row at every phase boundary using **independent psycopg2 transactions** so the audit log survives even if the pipeline crashes mid-flight.

### Phase 1 — Understand (`data/`)
- [`ingestion.py`](src/ive/data/ingestion.py) loads CSV/Parquet, infers column types, validates schema, persists to artifact store.
- [`profiler.py`](src/ive/data/profiler.py) computes per-column statistics, missingness, target correlations, and a quality score.
- [`preprocessor.py`](src/ive/data/preprocessor.py) standardizes numeric features and encodes categoricals.
- [`validator.py`](src/ive/data/validator.py) flags zero-variance columns and target issues.
- A holdout is split off; if a `time_column` is configured, the split is **temporal** (older→train, newer→holdout).

### Phase 2 — Model (`models/`)
- [`cross_validator.py`](src/ive/models/cross_validator.py) runs K-fold CV (default K=5).
- For each fold, both [`linear_model.py`](src/ive/models/linear_model.py) and [`xgboost_model.py`](src/ive/models/xgboost_model.py) are fit.
- Out-of-fold predictions and residuals are collected and persisted (`Residual` table).
- Per-fold metrics (`TrainedModel` rows) capture train/val score and hyperparams.
- SHAP values are computed via [`shap_interactions.py`](src/ive/detection/shap_interactions.py) — `TreeExplainer` for XGBoost, `KernelExplainer` for linear (capped at `shap_sample_size`).

### Phase 3 — Detect (`detection/`)
- [`subgroup_discovery.py`](src/ive/detection/subgroup_discovery.py) — for every column, bin values (quantile bins for numeric, per-value for categorical) and run a two-sample Kolmogorov-Smirnov test on the residual distribution inside vs outside the subgroup. Apply Benjamini-Hochberg FDR correction. Filter by minimum Cohen's *d*.
- [`clustering.py`](src/ive/detection/clustering.py) — take the worst 20% of |residuals|, scale features, fit HDBSCAN. Each surviving cluster becomes a candidate "geometric" pattern with a center and radius.
- [`shap_interactions.py`](src/ive/detection/shap_interactions.py) — extract top-k feature interaction pairs from SHAP for "interaction" patterns.
- [`temporal_analysis.py`](src/ive/detection/temporal_analysis.py) — only when a datetime column exists. Detects trend (Kendall's τ), seasonality (binned residuals by period), and regime shifts.
- [`pattern_scorer.py`](src/ive/detection/pattern_scorer.py) ranks all candidates by `0.60·effect_size + 0.40·coverage`, deduplicates by Jaccard ≥ 0.9, returns top-k.

### Phase 4 — Construct (`construction/`)
- [`variable_synthesizer.py`](src/ive/construction/variable_synthesizer.py) — turns each pattern into a concrete construction rule. Subgroups → binary indicator; clusters → continuous score `1 / (1 + d)` from cluster center; interactions → product / threshold rules.
- [`bootstrap_validator.py`](src/ive/construction/bootstrap_validator.py) — for each candidate, draw N bootstrap resamples, re-apply the rule, and verify three gates simultaneously: variance floor, score range, support rate window. The fraction of surviving resamples is the **bootstrap presence rate**. Demo: ≥0.50 / Production: ≥0.70.
- [`causal_checker.py`](src/ive/construction/causal_checker.py) — flag confounding-proxy candidates (high partial correlation with existing features) and reverse-causality candidates (high correlation with target). Does not delete; reduces `confidence_score` and emits warnings.
- [`explanation_generator.py`](src/ive/construction/explanation_generator.py) — converts every accepted/rejected variable into business-readable prose with hedged, non-causal language (effect-size buckets: large/medium/small/negligible).

### Phase 5 — Evaluate (optional, requires holdout)
- Baseline model trained on `X_train` only.
- Greedy forward selection of validated latent variables: at each step, pick the LV that maximizes holdout R² gain.
- Records marginal improvement per LV and a final "ensemble uplift" metric on the experiment.

---

## 6. Repository Layout

```
invisible-variables-engine/
├── AGENTS.md                       ← this document
├── README.md                       ← user-facing overview
├── LICENSE                         ← MIT
├── PROJECT_RULES.md                ← GSD methodology rules
├── GSD-STYLE.md                    ← GSD doc style reference
├── Makefile                        ← developer workflow targets
├── pyproject.toml                  ← Poetry + ruff + mypy + pytest config
├── poetry.lock                     ← deterministic dep lock
├── poetry.toml                     ← local Poetry config
├── pyrightconfig.json              ← Pyright/Pylance settings (relaxed)
├── alembic.ini                     ← Alembic config (DB migrations)
├── docker-compose.yml              ← multi-service stack
├── Dockerfile                      ← multi-stage (api / worker / streamlit)
├── .dockerignore                   ← excludes from build context
├── .env.example                    ← env-var template (copy to .env)
├── .env                            ← local config (gitignored)
├── .gitignore                      ← VCS ignores
├── .pre-commit-config.yaml         ← pre-commit hooks (ruff, mypy)
├── model_capabilities.yaml         ← LLM model selection registry
│
├── src/ive/                        ← THE PYTHON PACKAGE (see §7)
│   ├── api/                        ← FastAPI layer
│   │   ├── middleware/             ← auth, error, rate limit
│   │   ├── v1/                     ← versioned routes + schemas
│   │   └── websocket/              ← live progress streaming
│   ├── auth/                       ← scopes, key utils, egress checks (PR-2/3)
│   ├── config.py                   ← Pydantic Settings master
│   ├── construction/               ← Phase 4
│   ├── core/                       ← pipeline orchestrator
│   ├── data/                       ← Phase 1
│   ├── db/                         ← SQLAlchemy models + repos
│   ├── detection/                  ← Phase 3
│   ├── llm/                        ← Groq / OpenAI-compat enrichment (PR-1)
│   ├── main.py                     ← FastAPI factory
│   ├── models/                     ← Phase 2 (ML models)
│   ├── storage/                    ← artifact persistence
│   ├── utils/                      ← logging, stats, helpers, reporting
│   └── workers/                    ← Celery app + tasks
│
├── streamlit_app/                  ← visual workbench (see §12)
│   ├── app.py
│   ├── components/                 ← charts, sidebar, theme, widgets
│   └── pages/                      ← 01_upload → 05_compare
│
├── alembic/                        ← DB migrations (see §15)
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 20260302_1958-c25bd1018dab_initial_schema.py
│
├── scripts/                        ← utility CLIs (see §7.12)
│   ├── benchmark_ive.py
│   ├── calibrate_demo_datasets.py
│   ├── generate_demo_datasets.py
│   ├── generate_synthetic_data.py
│   ├── seed_db.py
│   └── search_repo.{sh,ps1}, setup_search.{sh,ps1},
│       validate-{all,skills,templates,workflows}.{sh,ps1}
│
├── tests/                          ← see §13
│   ├── conftest.py
│   ├── unit/                       (~150 tests)
│   ├── integration/                (~30 tests; needs services)
│   ├── statistical/                (numerical correctness)
│   ├── e2e/                        (full Docker workflow)
│   └── fixtures/                   (synthetic data + demo CSVs)
│
├── demo_datasets/                  ← 5 + 1 demo CSVs with metadata.json
├── docs/                           ← 17 reference documents (see §16)
├── benchmark_results/              ← persisted benchmark outputs
│
├── adapters/                       ← model-agnostic agent prompts
│   ├── CLAUDE.md
│   ├── GEMINI.md
│   └── GPT_OSS.md
│
├── .github/workflows/ci.yml        ← GitHub Actions pipeline
├── .gsd/                           ← GSD methodology project state
│   ├── ARCHITECTURE.md, STACK.md, STATE.md, DEBUG.md
│   ├── examples/  templates/
├── .agent/                         ← agent skills + workflows
│   ├── skills/  workflows/
├── .streamlit/                     ← Streamlit theme/server config
├── .vscode/                        ← editor settings
├── .gemini/                        ← Gemini agent settings
├── .claude/                        ← Claude Code settings
└── .venv/ .pytest_cache/ .mypy_cache/ .ruff_cache/   ← caches (gitignored)
```

---

## 7. Source Code Reference

### 7.1 `src/ive/main.py`
FastAPI app factory. Builds the application object, attaches `auth`, `error_handler`, and `rate_limit` middleware, mounts the v1 router under `/api/v1`, configures structlog, and registers Sentry if `SENTRY_DSN` is set.

### 7.2 `src/ive/config.py`
Single source of truth for runtime configuration. Built from eight Pydantic-Settings sub-classes (`DatabaseSettings`, `RedisSettings`, `CelerySettings`, `SecuritySettings`, `MLSettings`, `DetectionSettings`, `LLMSettings`, `StorageSettings`) merged into one `Settings` class via multiple inheritance. `LLMSettings` carries Groq / OpenAI-compatible client config plus cache, breaker, and budget controls (added in PR-1; off by default until staging soak).

Key behaviors:
- Resolution order: env vars → `.env` → field defaults.
- `database_url` validator auto-rewrites `postgresql://` → `postgresql+asyncpg://`.
- `@model_validator(mode="after")` enforces production strictness: `SECRET_KEY` ≥ 32 chars, `VALID_API_KEYS` non-empty, warns on `DEBUG=true`.
- `get_settings()` is `@lru_cache`'d — call `.cache_clear()` in tests.
- `__repr__` redacts secrets to prevent leakage in logs.

### 7.3 `src/ive/core/pipeline.py`
The `IVEPipeline` orchestrator. See §5 for phase semantics. Key engineering details:
- All event-log writes go through a side-channel sync psycopg2 connection so audit events persist even if the main async session is in an aborted transaction.
- Each phase emits `phase_started` / `phase_completed` / `phase_failed` events with structured payloads.
- Holdout split is deterministic given `random_seed`; temporal split is used when a `time_column` is configured.

### 7.4 `src/ive/api/`

#### `api/middleware/auth.py`
API-key validation — checks `X-API-Key` header against `Settings.api_keys_set`. Whitelists `/health*`, `/docs`, `/openapi.json`, and WebSocket paths.

#### `api/middleware/error_handler.py`
Catches all unhandled exceptions, logs with structlog at error level, returns shape: `{"error": {"code": "...", "message": "...", "request_id": "..."}}`.

#### `api/middleware/rate_limit.py`
SlowAPI-based per-IP sliding window. Window and request count come from `RATE_LIMIT_*` settings.

#### `api/v1/router.py`
Aggregates the four endpoint routers (datasets, experiments, latent_variables, health) into one `APIRouter` mounted at `/api/v1`.

#### `api/v1/dependencies.py`
FastAPI dependencies: `get_db()` yields async session, `get_pagination()` parses `?skip=&limit=` query params with safe defaults.

#### `api/v1/endpoints/`
Per-resource route files. See §8 for the full HTTP table.

#### `api/v1/schemas/`
- `dataset_schemas.py` — `DatasetResponse`, `DatasetListResponse`, `DatasetProfileResponse`.
- `experiment_schemas.py` — `ExperimentCreate`, `ExperimentResponse`, `ExperimentListResponse`, `ExperimentProgressResponse`, `ExperimentEventResponse`, `ExperimentEventsListResponse`, `ErrorPatternResponse`, `ExperimentSummaryResponse`.
- `latent_variable_schemas.py` — `LatentVariableResponse`, `LatentVariableListResponse`, application input/output models.

#### `api/websocket/progress.py`
WebSocket endpoint for real-time progress streaming. Streamlit's monitor page subscribes here.

### 7.5 `src/ive/data/`
- **`ingestion.py`** — `DataIngestionService.ingest()`: parse → infer types → validate → persist.
- **`profiler.py`** — `DataProfiler.profile()`: per-column stats, missingness, top correlations, quality score 0-100.
- **`preprocessor.py`** — `DataPreprocessor.transform()`: StandardScaler + categorical encoders → numeric matrix.
- **`validator.py`** — `DataValidator.validate()`: warning + dropped-columns report.

### 7.6 `src/ive/models/`
- **`base_model.py`** — `IVEModel` ABC with `fit / predict / get_shap_values`.
- **`linear_model.py`** — `LinearIVEModel` (sklearn LinearRegression + KernelExplainer SHAP).
- **`xgboost_model.py`** — `XGBoostIVEModel` (XGBRegressor + TreeExplainer SHAP).
- **`cross_validator.py`** — `CrossValidator.run(model_factory, X, y, k)`: returns CV result with mean/std score, fold ids, OOF preds & residuals.
- **`residual_analyzer.py`** — placeholder for heteroscedasticity / outlier diagnostics (not yet wired).

### 7.7 `src/ive/detection/`
| Module                    | Class                          | Output pattern type              |
| ------------------------- | ------------------------------ | -------------------------------- |
| `subgroup_discovery.py`   | `SubgroupDiscovery`            | `subgroup`                       |
| `clustering.py`           | `HDBSCANClustering`            | `cluster`                        |
| `shap_interactions.py`    | `SHAPInteractionAnalyzer`      | `interaction`                    |
| `temporal_analysis.py`    | `TemporalAnalyzer`             | `temporal_trend`, `seasonality`  |
| `pattern_scorer.py`       | `PatternScorer`                | (ranks, dedupes, doesn't create) |

All detectors return a uniform `Pattern` shape with `effect_size`, `coverage`, `p_value`, `p_value_adjusted`, `metadata`.

### 7.8 `src/ive/construction/`
- **`variable_synthesizer.py`** — `VariableSynthesizer.synthesize(patterns)` → list of `LatentVariableCandidate` with `construction_rule` JSON.
- **`bootstrap_validator.py`** — `BootstrapValidator.validate(candidates, X, y)`: returns each candidate annotated with `bootstrap_presence_rate`, `status`, `rejection_reason`.
- **`causal_checker.py`** — `CausalChecker.check(candidates, X, y)`: appends causal warnings, lowers confidence.
- **`explanation_generator.py`** — `ExplanationGenerator.explain_pattern / explain_variable / explain_experiment`: produces business-friendly prose for every artifact.

### 7.9 `src/ive/db/`
- **`database.py`** — async SQLAlchemy engine + session factory + `get_session()` context manager.
- **`models.py`** — see §9 for full schema.
- **`repositories/base_repo.py`** — generic CRUD over any ORM model (typed via TypeVar).
- **`repositories/dataset_repo.py`** — `create / get_by_id / search_by_name / get_all / count / update / delete`.
- **`repositories/experiment_repo.py`** — adds `mark_started / mark_completed / mark_failed / mark_cancelled / update_progress / add_trained_model / add_residuals_batch / add_error_patterns_batch / get_error_patterns / get_events`.
- **`repositories/latent_variable_repo.py`** — `bulk_create / get_by_experiment / get_by_id / filter by status`.
- **`repositories/api_key_repo.py`** — multi-user auth: `create / get_by_hash / list / revoke / record_usage` (PR-2).
- **`repositories/dataset_column_metadata_repo.py`** — per-column sensitivity: `list_for_dataset / bulk_create_default / bulk_set / public_column_names` (PR-3).
- **`repositories/llm_explanation_repo.py`** — LLM column writers: `set_lv_explanation / bulk_mark_lvs_disabled / set_experiment_explanation / mark_experiment_disabled` (PR-4).

### 7.10 `src/ive/storage/`
- **`artifact_store.py`** — `ArtifactStore` factory + `LocalArtifactStore` (aiofiles) + `S3ArtifactStore` (optional, requires `boto3` extra). Methods: `save_file / load_file / delete_file`.

### 7.11 `src/ive/utils/`
- **`logging.py`** — structlog configuration; JSON output with bound context (request_id, experiment_id, etc.).
- **`statistics.py`** — `cohens_d`, `pooled_std`, `bh_correction`, percentile helpers.
- **`reporting.py`** — `build_full_report()` (JSON), `patterns_to_csv()`, `latent_variables_to_csv()`.
- **`helpers.py`** — UUID parsing, type coercions, JSON-safe numpy serializers (the recent `numpy bool_/int_/float_` JSON fix lives here).

### 7.12 `src/ive/workers/`
- **`celery_app.py`** — Celery factory: broker/result backend wired from settings; queues `default`, `analysis`, `high_priority`; JSON serializer; result TTL = `celery_result_expires`.
- **`tasks.py`** — see §10.
- **`llm_enrichment.py`** — async core of `generate_llm_explanations`. Flag-off short-circuits to mark every LV `disabled`; flag-on builds an `httpx.AsyncClient` + best-effort Redis cache + breaker, runs sem-bounded per-LV gather through `generate_with_fallback`, then experiment-level headline + narrative. Cancel-event polled at LV boundaries (PR-4).

### 7.13 `src/ive/llm/`
The LLM enrichment package — Groq / OpenAI-compatible. **Off by default** (`LLM_EXPLANATIONS_ENABLED=false`); when on, post-processes pipeline output to produce business-readable prose with strict hallucination guardrails and rule-based fallback. Added in PR-1.

| Module | Responsibility |
|---|---|
| `client.py` | `GroqClient` async chat-completions wrapper. Retry with backoff on 429/5xx, `Retry-After` honored. Distinguishes `LLMUnavailable` / `LLMBadRequest` / `LLMAuthError` so the breaker only counts service-health failures. |
| `prompts.py` | Versioned `(name, version)` template registry. Five templates: `lv_explanation`, `pattern_summary`, `experiment_headline`, `experiment_narrative`, `recommendations`. `template_sha()` feeds the cache key so structural template edits auto-invalidate. |
| `validators.py` | `composite_validate()` chain: length sanity → injection-echo → banned phrases → numeric grounding (±2% tolerance, pairwise derivations). `sanitize_user_input()` strips injection markers + truncates free-text. |
| `cache.py` | `RedisLLMCache` with per-entity index sets so dataset/experiment delete cascades to cache keys. |
| `circuit_breaker.py` | Redis-backed consecutive-failure counter with cooldown TTL. |
| `fallback.py` | `generate_with_fallback(...)` orchestrator: flag-check → breaker → cache → client → validate → cache.set, with graceful degradation on every failure path. |
| `payloads.py` | `build_lv_payload` + `build_experiment_payload` — egress-aware fact extraction; non-public columns never appear in the output. |
| `rule_based.py` | Adapters wrapping `ExplanationGenerator` into the `Callable[[], str]` shape `generate_with_fallback` expects. |
| `types.py` | Pydantic models: `GenerationRequest`, `GenerationResult`, `ValidationReport`. |

### 7.14 `src/ive/auth/`
Multi-user auth + scopes + LLM data-egress checks. Added in PR-2 + PR-3.

| Module | Responsibility |
|---|---|
| `scopes.py` | `Scope` enum (`read | write | admin`), `AuthContext` dataclass, `require_scope(...)` FastAPI dependency. |
| `utils.py` | `generate_api_key()`, `hash_api_key()` (bcrypt). |
| `egress.py` | `evaluate_lv_egress(referenced, public)` — binary public/non-public sensitivity gate. Blocks LVs whose construction rule references *any* non-public column. `filter_payload_columns()` is defense-in-depth. |

### 7.15 `scripts/`
| Script                          | Purpose                                                          |
| ------------------------------- | ---------------------------------------------------------------- |
| `generate_synthetic_data.py`    | Build synthetic regression/classification CSVs.                  |
| `generate_demo_datasets.py`     | Build the 5 named demo CSVs with embedded ground-truth signals.  |
| `calibrate_demo_datasets.py`    | Tune demo signals so they're discoverable at expected effect sizes. |
| `seed_db.py`                    | Seed dev DB with demo datasets + sample experiments.             |
| `benchmark_ive.py`              | Sweep dataset sizes; record runtime, memory, detection quality.  |
| `search_repo.{sh,ps1}`          | Bootstrap helper: ripgrep-based code search shortcuts.           |
| `setup_search.{sh,ps1}`         | Install the search shortcuts into shell rc files.                |
| `validate-{all,skills,templates,workflows}.{sh,ps1}` | GSD doc-validation runners (see `.gsd/`). |

---

## 8. REST API Reference

All endpoints are mounted under `/api/v1`. Authentication: `X-API-Key: <key>` (defaults `dev-key-1`, `dev-key-2`).

### 8.1 Health
| Method | Path                  | Purpose                                                              |
| ------ | --------------------- | -------------------------------------------------------------------- |
| GET    | `/health`             | Liveness — 200 if process is up. No dependency checks.               |
| GET    | `/health/ready`       | Readiness — 200 only if Postgres + Redis reachable; 503 otherwise.   |

### 8.2 Datasets — [`endpoints/datasets.py`](src/ive/api/v1/endpoints/datasets.py)
| Method | Path                                          | Purpose                                                       |
| ------ | --------------------------------------------- | ------------------------------------------------------------- |
| POST   | `/datasets/`                                  | Multipart upload (CSV + `target_column`). Validates, ingests, profiles, **seeds per-column sensitivity rows at `non_public`** (PR-3), returns `DatasetResponse`. |
| GET    | `/datasets/`                                  | List datasets (paginated, optional `?name=`).                 |
| GET    | `/datasets/{dataset_id}`                      | Full detail incl. schema and column info.                     |
| GET    | `/datasets/{dataset_id}/profile`              | Stats profile, correlations, quality score.                   |
| GET    | `/datasets/{dataset_id}/columns/`             | List per-column sensitivity (read scope; PR-3).               |
| PUT    | `/datasets/{dataset_id}/columns/`             | Bulk-update column sensitivity (write scope; PR-3).           |
| DELETE | `/datasets/{dataset_id}`                      | Delete row + artifact file (204). Cascades to column metadata + experiments. |

### 8.3 Experiments — [`endpoints/experiments.py`](src/ive/api/v1/endpoints/experiments.py)
| Method | Path                                                       | Purpose                                                      |
| ------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| POST   | `/experiments/`                                            | Create + queue Celery task. Returns 201 with `ExperimentCreateResponse`. |
| GET    | `/experiments/`                                            | List with filters `?status=&dataset_id=`.                    |
| GET    | `/experiments/compare`                                     | Side-by-side diff of 2+ experiments (config + LV overlap).   |
| GET    | `/experiments/{id}`                                        | Full detail.                                                 |
| GET    | `/experiments/{id}/progress`                               | Lightweight poll: `{percentage, stage, status}`.             |
| GET    | `/experiments/{id}/events`                                 | Chronological audit log.                                     |
| GET    | `/experiments/{id}/patterns`                               | All discovered patterns.                                     |
| GET    | `/experiments/{id}/latent-variables`                       | LVs (filterable by `?status=`).                              |
| GET    | `/experiments/{id}/summary`                                | Compact executive summary (headline + top findings).         |
| GET    | `/experiments/{id}/report`                                 | Full JSON report (experiment + dataset + patterns + LVs).    |
| GET    | `/experiments/{id}/patterns/export`                        | CSV stream of patterns.                                      |
| GET    | `/experiments/{id}/latent-variables/export`                | CSV stream of LVs.                                           |
| POST   | `/experiments/{id}/cancel`                                 | Revoke Celery task; mark cancelled.                          |
| DELETE | `/experiments/{id}`                                        | Cascade delete experiment and all children.                  |

### 8.4 Latent Variables — [`endpoints/latent_variables.py`](src/ive/api/v1/endpoints/latent_variables.py)
| Method | Path                                  | Purpose                                                                |
| ------ | ------------------------------------- | ---------------------------------------------------------------------- |
| GET    | `/latent-variables/`                  | List across experiments (`?status=&experiment_id=`).                   |
| GET    | `/latent-variables/{id}`              | Full detail incl. construction rule and stability metrics.             |
| POST   | `/latent-variables/apply`             | Apply selected validated LVs to a new uploaded CSV; return scores.     |

### 8.5 WebSocket
- `WS /api/v1/ws/progress/{experiment_id}` — real-time progress events. Sends `{stage, percentage, event_type, payload}` as the worker emits events.

### 8.6 API Keys (admin scope) — [`endpoints/api_keys.py`](src/ive/api/v1/endpoints/api_keys.py)
Multi-user auth admin surface. Added in PR-2.

| Method | Path                          | Purpose                                                                  |
| ------ | ----------------------------- | ------------------------------------------------------------------------ |
| POST   | `/api-keys/`                  | Create a new API key with scopes; returns the raw key **once** (the hash is what's persisted). |
| GET    | `/api-keys/`                  | List existing keys (admin scope only).                                   |
| DELETE | `/api-keys/{api_key_id}`      | Revoke (sets `is_active=false`).                                         |

### 8.7 LLM-explanation status fields (cross-cutting)
Added in PR-5: every endpoint that returns an explanation also returns `explanation_source: "llm"|"rule_based"`, `llm_explanation_pending: bool`, `llm_explanation_status: "pending"|"ready"|"failed"|"disabled"`. See `docs/RESPONSE_CONTRACT.md` §4 for the full state table.

Affected endpoints:
- `GET /experiments/{id}/summary`
- `GET /experiments/{id}/latent-variables`
- `GET /latent-variables/`
- `GET /latent-variables/{id}`

---

## 9. Database Schema

Defined in [`src/ive/db/models.py`](src/ive/db/models.py). All tables use UUID primary keys, `created_at`/`updated_at` timestamps, and proper FKs with cascade rules.

| Table                       | Class                     | Purpose                                                                                  |
| --------------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| `datasets`                  | `Dataset`                 | Uploaded CSV/Parquet metadata: file_path, checksum, schema_json, row/col counts, target_column. |
| `experiments`               | `Experiment`              | Analysis run lifecycle: status, progress_percentage, current_stage, config (JSONB), uplift metrics. **+ LLM cols** (`llm_headline`, `llm_narrative`, `llm_recommendations`, `llm_explanation_status`, `llm_task_id`, etc. — PR-1). |
| `trained_models`            | `TrainedModel`            | Per-fold model metrics: model_type, fold_number, train_score, val_score, hyperparams.    |
| `residuals`                 | `Residual`                | OOF predictions and residuals: row_index, predicted_value, residual_value, fold.         |
| `error_patterns`            | `ErrorPattern`            | Detected patterns: pattern_type (subgroup/cluster/interaction/temporal), effect_size, p_value, p_value_adjusted, definition (JSONB). |
| `latent_variables`          | `LatentVariable`          | Constructed LVs: construction_rule (JSONB), bootstrap_presence_rate, status, rejection_reason, confidence_score, explanation. **+ LLM cols** (`llm_explanation`, `llm_explanation_status`, etc. — PR-1). |
| `experiment_events`         | `ExperimentEvent`         | Append-only audit log: phase, event_type, payload (JSONB), timestamp.                    |
| `api_keys`                  | `APIKey`                  | Hashed API keys with scopes and rate limits. Multi-user model in PR-2.                  |
| `auth_audit_log`            | `AuthAuditLog`            | Per-request auth audit: api_key_id, route, status, IP, timestamp (PR-2).                |
| `dataset_column_metadata`   | `DatasetColumnMetadata`   | Per-column sensitivity (`public` / `non_public`). Default `non_public` on upload (PR-3). |
| `explanation_feedback`      | `ExplanationFeedback`     | Thumbs-up/down on generated explanations; carries `prompt_version` + `model_version` (PR-1). |

Initial migration: [`alembic/versions/20260302_1958-c25bd1018dab_initial_schema.py`](alembic/versions/20260302_1958-c25bd1018dab_initial_schema.py).

**Phase A migrations** (per `docs/RESPONSE_CONTRACT.md` §10):
- `add_lv_llm_columns` — code-first
- `add_experiment_llm_columns` — code-first
- `add_explanation_feedback_table` — code-first
- `add_dataset_column_metadata` — schema-first
- `extend_api_keys_for_multi_user` + `add_auth_audit_log` — same-release

---

## 10. Background Jobs

[`src/ive/workers/tasks.py`](src/ive/workers/tasks.py)

| Task                                    | Trigger                     | Behavior                                                                                                    |
| --------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `run_experiment(experiment_id, config)` | `POST /experiments/`        | Bridges sync Celery → async `IVEPipeline` via `asyncio.run()`. Updates progress, persists artifacts. **Chains `generate_llm_explanations` on success** (PR-4). Returns `{n_patterns, n_validated, elapsed_seconds}`. |
| `profile_dataset(dataset_id, file_path)`| Post-upload (auto)          | Loads file, runs `DataProfiler`, merges profile into `schema_json`.                                         |
| `cancel_experiment(task_id, exp_id)`    | `POST /experiments/{id}/cancel` | Revokes the Celery task with SIGTERM, marks experiment `cancelled`.                                     |
| `generate_llm_explanations(experiment_id)` | Chained from `run_experiment` | `AbortableTask`-based; polls `is_aborted()` between LVs. Flag-off → mark every LV `disabled`. Flag-on → sem-bounded gather through `generate_with_fallback`, then experiment-level headline + narrative. Egress-checked per LV (PR-4). |
| `health_check_task()`                   | `/health/ready`             | Lightweight echo to verify worker responsiveness.                                                            |

Queues: `default`, `analysis`, `high_priority`. Run a worker with all three: `celery -A ive.workers.celery_app worker -Q default,analysis,high_priority`.

---

## 11. Configuration

`.env.example` is the authoritative template. Sections:

| Section                  | Vars                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| Application              | `ENV`, `DEBUG`, `LOG_LEVEL`, `APP_NAME`, `APP_VERSION`, `API_PORT`                                     |
| Security                 | `SECRET_KEY`, `API_KEY_HEADER`, `VALID_API_KEYS`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`           |
| Database (Postgres)      | `DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`, `DATABASE_POOL_TIMEOUT`, `POSTGRES_*`   |
| Redis                    | `REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_DB`                                  |
| Celery                   | `CELERY_CONCURRENCY`, `CELERY_TASK_SERIALIZER`, `CELERY_RESULT_EXPIRES`, `CELERY_MAX_TASKS_PER_CHILD`  |
| Flower                   | `FLOWER_PORT`, `FLOWER_USER`, `FLOWER_PASSWORD`                                                        |
| Streamlit                | `STREAMLIT_PORT`, `API_BASE_URL`                                                                        |
| ML Defaults              | `RANDOM_SEED`, `DEFAULT_CV_FOLDS`, `DEFAULT_TEST_SIZE`, `MAX_FEATURES`, `MIN_CLUSTER_SIZE`, `SHAP_SAMPLE_SIZE` |
| Detection                | `DEMO_MODE`, `DEFAULT_STABILITY_THRESHOLD`, `DEMO_STABILITY_THRESHOLD`, `DEFAULT_MIN_EFFECT_SIZE`, `DEFAULT_MIN_SUBGROUP_SIZE`, `DEFAULT_MIN_VARIANCE_THRESHOLD` |
| Storage                  | `ARTIFACT_STORE_TYPE` (`local` / `s3`), `ARTIFACT_BASE_DIR`, `S3_BUCKET_NAME`, `AWS_*`, `S3_ENDPOINT_URL`, `MAX_UPLOAD_SIZE_MB` |
| Observability            | `SENTRY_DSN`, `ENABLE_METRICS`                                                                          |
| LLM (Groq / OpenAI-compat) | `LLM_EXPLANATIONS_ENABLED` (off by default), `LLM_SELF_HOSTED_MODE`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `GROQ_TIMEOUT_SECONDS`, `GROQ_MAX_RETRIES`, `GROQ_MAX_OUTPUT_TOKENS`, `GROQ_TEMPERATURE` (default 0.0), `GROQ_MAX_CONCURRENCY`, `LLM_CACHE_TTL_SECONDS`, `LLM_REDIS_DB` (default 2 — separate from broker/results), `LLM_PROMPT_VERSION`, `LLM_CIRCUIT_BREAKER_THRESHOLD`, `LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS`, `LLM_VALIDATION_RETRY_MAX`, `LLM_BATCH_SIZE_LVS`, `LLM_BATCH_TOKEN_SAFETY_THRESHOLD`, `LLM_DAILY_TOKEN_BUDGET`, `LLM_PER_EXPERIMENT_TOKEN_CAP`, `LLM_DATA_EGRESS_DEFAULT` (default `deny`), `LLM_VALIDATOR_PROFILE` |

### Demo vs Production thresholds (controlled by `analysis_mode`)

| Aspect                            | Demo  | Production |
| --------------------------------- | ----- | ---------- |
| Bootstrap stability threshold     | 0.50  | 0.70       |
| Min variance floor                | 1e-7  | 1e-5       |
| Min score range                   | 0.01  | 0.05       |
| Support rate window               | 0.5%–98% | 1%–95%  |
| Subgroup min samples              | 20    | 30         |
| Effect size threshold             | 0.15  | 0.20       |

Use **demo** for synthetic datasets / demos. Use **production** for real-world analyses where false positives carry operational cost.

---

## 12. Streamlit UI

Entry: [`streamlit_app/app.py`](streamlit_app/app.py). Built around the multi-page Streamlit pattern.

| Page                                    | Purpose                                                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `pages/01_upload.py`                    | CSV uploader, target column selector, optional time column, data preview, ingest trigger.                     |
| `pages/02_configure.py`                 | Pick dataset, set CV folds, test size, model types, analysis mode (demo/production), bootstrap iterations.    |
| `pages/03_monitor.py`                   | Real-time progress bar, stage display, WebSocket subscription, lifecycle event log.                           |
| `pages/04_results.py`                   | Patterns + LVs with stability/importance, business explanations, rejection reasons, CSV download. **AI-assisted badges + dataset-level sensitivity banner** (PR-6). |
| `pages/05_compare.py`                   | Pick 2+ experiments → config diff, pattern overlap, LV overlap (common / unique-A / unique-B).                |
| `pages/06_dataset_settings.py`          | Per-column sensitivity editor: bulk actions, diff preview, save (PR-7).                                       |

### Components
- `components/sidebar.py` — global nav + API connectivity badge.
- `components/theme.py` — Carbon Design System theming (dark by default). Adds `explanation_source_badge()` (PR-6) for AI-assisted / pending / failed / disabled states.
- `components/charts.py` — Plotly visualisations (effect-size hist, p-value hist, LV importance, timeline).
- `components/widgets.py` — metric cards, status badges, spinners, data tables.

Run locally: `make streamlit-local` or via Docker: `make dev` → http://localhost:8501.

---

## 13. Test Suite

Configured in [`pyproject.toml`](pyproject.toml) (`[tool.pytest.ini_options]`). Run with `make test*`.

| Tier          | Path                  | Count | Needs                  | Run                      |
| ------------- | --------------------- | ----- | ---------------------- | ------------------------ |
| Unit          | `tests/unit/`         | ~150  | Nothing (in-process)   | `make test-unit`         |
| Integration   | `tests/integration/`  | ~30   | Postgres + Redis up    | `make test-integration`  |
| Statistical   | `tests/statistical/`  | ~20   | Nothing                | `make test-statistical`  |
| End-to-end    | `tests/e2e/`          | small | Full Docker stack      | `pytest tests/e2e -m e2e` |

### Notable test files
- `tests/conftest.py` — shared fixtures (DB session, Celery eager mode, mock artifact store).
- `tests/fixtures/synthetic_datasets.py` — programmatic regression/classification generators.
- `tests/fixtures/demo_csv_files.py` — CSV fixture builders for upload tests.
- `tests/unit/test_bootstrap_validator.py` — triple-gate logic.
- `tests/unit/test_subgroup_discovery.py` — KS test + BH correction correctness.
- `tests/unit/test_pipeline_integration_smoke.py` — fast end-to-end smoke without DB.
- `tests/integration/test_pipeline_e2e.py` — full pipeline against real DB.
- `tests/integration/test_job_processing.py` — Celery task queueing.
- `tests/statistical/test_false_positives.py` — FPR validation on `no_hidden_random_noise` data.
- `tests/statistical/test_ground_truth.py` — recall on demo datasets with known LVs.
- `tests/statistical/test_reproducibility.py` — deterministic results given fixed seed.
- `tests/e2e/test_full_workflow.py` — upload → experiment → monitor → results.

Markers (declared in `pyproject.toml`): `slow`, `integration`, `e2e`, `statistical`, `unit`.

---

## 14. Demo Datasets

Located in `demo_datasets/`. Each CSV has a sibling `<name>.metadata.json` describing its ground-truth latent variable.

| Dataset                            | Target            | Hidden Variable                                            | Affected | Detection Type |
| ---------------------------------- | ----------------- | ---------------------------------------------------------- | -------- | -------------- |
| `delivery_hidden_weather.csv`      | `delivery_time`   | Storm delay zone (distance > 10 & traffic > 7.5)           | 19.5%    | Subgroup       |
| `healthcare_hidden_risk.csv`       | `recovery_days`   | Post-surgery complication (BMI > 30 & BP > 150)            | 22.3%    | Subgroup       |
| `manufacturing_hidden_shift.csv`   | `defect_rate`     | Night shift instability (vibration > 7 & humidity > 75)    | 10.7%    | Subgroup       |
| `retail_hidden_promo.csv`          | `spend_amount`    | Premium promo eligibility (loyalty > 0.8 & basket > 8)     | 8.7%     | Subgroup       |
| `customer_churn_hidden_signal.csv` | (per metadata)    | Churn driver (per metadata)                                | varies   | Subgroup       |
| `no_hidden_random_noise.csv`       | `target`          | None (negative-control dataset)                            | 0%       | —              |

The `no_hidden_random_noise` dataset is critical for FPR validation in `tests/statistical/test_false_positives.py`.

---

## 15. Operations

### 15.1 Dockerfile (multi-stage)
```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────┐
│  base   │─▶│   api   │  │ worker  │  │ streamlit  │
│ Py 3.11 │  │ uvicorn │  │ celery  │  │  streamlit │
│ Poetry  │  │  :8000  │  │  -Q ... │  │   :8501    │
└─────────┘  └─────────┘  └─────────┘  └────────────┘
```
- **base** — `python:3.11-slim`, system deps (libpq, build tools), Poetry, locked Python deps via `poetry install --no-root --only main`.
- **api** — runs `alembic upgrade head` then `uvicorn ive.main:app`. Healthcheck hits `/api/v1/health`.
- **worker** — runs `celery -A ive.workers.celery_app worker -Q default,analysis,high_priority`. Concurrency from `CELERY_CONCURRENCY`.
- **streamlit** — runs `streamlit run streamlit_app/app.py`. Headless mode.

### 15.2 docker-compose.yml
Brings up: `postgres` (16-alpine, healthcheck), `redis` (7-alpine, password-protected, AOF on), `api` (depends on healthy postgres+redis), `worker`, `flower` (mher/flower:2.0), `streamlit`. Volumes: `postgres_data`, `redis_data`, `artifact_data`. Network: `ive_net`.

### 15.3 GitHub Actions — `.github/workflows/ci.yml`
- **Lint job** — Ruff lint + format check; mypy strict mode.
- **Unit Tests job** — pytest + coverage upload.
- **Integration Tests job** — spins up `postgres:16` + `redis:7` service containers, runs `alembic upgrade head`, then integration tests.

### 15.4 Alembic
- Config: [`alembic.ini`](alembic.ini)
- Env: [`alembic/env.py`](alembic/env.py) — uses `Settings.sync_database_url` (psycopg2 driver, not asyncpg).
- Version naming: `YYYYMMDD_HHMM-<rev>_<msg>.py`
- Initial migration creates the full 8-table schema with FK cascades.
- Workflow: `make makemigrations MSG="..."` → review → `make migrate` → `make migration-current` to confirm.

### 15.5 Pre-commit (`.pre-commit-config.yaml`)
Hooks: `ruff` (lint + format), `mypy`. Install: `pre-commit install`.

---

## 16. Documentation Index

`docs/` contains the long-form references. Read these for deeper context.

| File                              | Topic                                                          |
| --------------------------------- | -------------------------------------------------------------- |
| `API.md`                          | Full REST API spec (companion to Swagger).                     |
| `API_USAGE_GUIDE.md`              | Client integration walkthroughs (Python, curl, JS).            |
| `ARCHITECTURE_OVERVIEW.md`        | System design narrative.                                       |
| `HLD.md` / `LLD.md`               | High-level / Low-level design (academic-defense format).       |
| `BUSINESS_VALUE.md`               | Use cases and ROI framing.                                     |
| `CAPABILITY_MATRIX.md`            | Feature-by-feature capability table.                           |
| `DEMO_GUIDE.md`                   | Step-by-step demo script.                                      |
| `DEPLOYMENT_GUIDE.md`             | Production deployment.                                         |
| `PRESENTATION_SCRIPT.md`          | Talking points for live presentations.                         |
| `TROUBLESHOOTING.md`              | Common errors and fixes.                                       |
| `USER_GUIDE.md`                   | End-user playbook (Streamlit + API).                           |
| `VIVA_QA.md`                      | Defense Q&A reference.                                         |
| `model-selection-playbook.md`     | When to use linear vs XGBoost vs both.                         |
| `runbook.md`                      | Operational runbook (incident response, rollback).             |
| `testing.md`                      | Testing strategy and tier definitions.                         |
| `token-optimization-guide.md`     | Context-window optimization for agent workflows.               |
| `RESPONSE_CONTRACT.md`            | **Authoritative API/data contract** — reproducibility, FPR, egress, deploy ordering, sensitivity model, model migration. Read first before trusting an IVE number. (PR-9) |

The `.gsd/` and `.agent/` directories contain GSD methodology artifacts (project state, skills, workflows, templates) — see `PROJECT_RULES.md` and `GSD-STYLE.md` for the methodology itself.

The `adapters/{CLAUDE,GEMINI,GPT_OSS}.md` files are model-specific prompt adapters used when running this project under different LLM agents.

---

## 17. Coding Conventions

### Python
- **Python 3.11**, type-annotated. `mypy --strict` runs in CI.
- **Ruff** is the only linter/formatter; config in `pyproject.toml` (`line-length=100`, `target-version="py311"`). Selected rules include pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade, annotations, bandit, naming.
- **Pydantic v2** for all data validation (request/response models, settings).
- **SQLAlchemy 2.0** style: `Mapped[...]` annotations + `mapped_column()`. Async sessions everywhere except Alembic.
- **Naming exceptions** for ML idioms: `X`, `y`, capital-X args/vars are allowed (`N802/N803/N806` ignored).
- **No bare `except:`** — always catch specifically and log via structlog.
- **Random seed** — `MLSettings.random_seed` (default 42) is the ONE place; pass it through, don't re-seed locally.

### Layered architecture (do not violate)
```
api ──▶ workers (enqueue only)
api ──▶ db.repositories ──▶ db.models
workers ──▶ core.pipeline ──▶ {data, models, detection, construction}
{data, models, detection, construction} ──▶ utils
```
- API endpoints **must not** import from `core/`, `detection/`, or `construction/` directly. They enqueue Celery tasks.
- ML modules **must not** import from `api/` or `db/repositories/`. They take dataframes/numpy in, return dataclasses out.
- DB writes from the pipeline go through repositories; the pipeline does not construct ORM objects directly.

### Logging & errors
- `structlog` JSON logs only. Bind context (`logger.bind(experiment_id=...)`) at the top of any task.
- Errors surface to the user via the global error handler middleware as `{"error": {"code", "message", "request_id"}}`.

### Tests
- New ML logic → unit test in `tests/unit/`.
- New endpoint → integration test in `tests/integration/`.
- New statistical claim ("we reject random noise at p<0.01") → statistical test.
- Use `pytest.mark.{unit, integration, statistical, e2e, slow}` markers.

### House rules from `PROJECT_RULES.md`
- **GSD methodology** — SPEC → PLAN → EXECUTE → VERIFY → COMMIT. Don't skip phases for non-trivial work.
- **Search first** — before writing any new helper/util, grep to see if it already exists.
- **No silent drops** — rejected latent variables get a `rejection_reason` and an explanation, never just `null`.
- **Audit log first** — any new pipeline phase must emit `phase_started`/`phase_completed`/`phase_failed` events.

---

## 18. Glossary

| Term                          | Definition                                                                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latent variable (LV)**      | An unmeasured factor whose effect is inferred from systematic patterns in model residuals. The output of IVE.                                                                       |
| **Pattern**                   | A statistically significant residual signature: a subgroup, a cluster, an interaction, or a temporal regime. Phase 3 produces these. Phase 4 turns them into LVs.                   |
| **OOF residual**              | "Out-of-fold" residual — the prediction error for a row when the model was trained on folds that did NOT include that row. Used because in-fold residuals are biased toward zero.   |
| **Effect size (Cohen's d)**   | Standardized mean difference of residuals inside vs outside a subgroup. >0.8 = large, 0.5 = medium, 0.2 = small, <0.15 = negligible (filtered out).                                  |
| **Bootstrap presence rate**   | The fraction of bootstrap resamples in which a candidate LV survives all three gates (variance, range, support). The stability metric.                                              |
| **Triple-gate**               | The three-test stability filter applied to every bootstrap resample: variance ≥ floor, range ≥ floor, support rate within window.                                                   |
| **Construction rule**         | A JSON-serializable specification of how to compute an LV from raw features. Persisted in `latent_variables.construction_rule`.                                                     |
| **Demo / Production mode**    | Two presets controlled by `analysis_mode`. Demo relaxes thresholds for synthetic data; Production tightens them for real data with cost of error.                                   |
| **Holdout uplift**            | Phase-5 measurement of how much R² (or analogous metric) improves on the holdout when the validated LVs are added to a baseline model. The business-facing "did it help?" number.   |
| **Rejection reason**          | Categorical label on a rejected LV: `low_presence_rate`, `low_variance`, `low_range`, `support_too_sparse`, `support_too_broad`, `causal_implausible`.                              |
| **Experiment event**          | An append-only audit row recording a phase boundary or significant pipeline action. Persisted via independent psycopg2 connection so it survives crashes.                           |

---

## 19. Troubleshooting

### Worker never picks up the task
```bash
make logs-worker            # confirm worker is consuming
docker compose ps redis     # confirm broker reachable
```
Check `CELERY_BROKER_URL` resolves to the right Redis db. Confirm worker started with `-Q default,analysis,high_priority`.

### `pg_isready` healthcheck fails
Make sure `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` in `.env` match what the healthcheck command expects. The defaults are `ive`/`ivepassword`/`ive_db`.

### `DATABASE_URL must start with 'postgresql'`
Pydantic validator caught a malformed DSN. Use `postgresql+asyncpg://...` or `postgresql://...` (the latter is auto-rewritten).

### Migrations fail with "target database is not up to date"
```bash
make migration-current
make migrate           # alembic upgrade head
```
If branched, run `alembic heads` to see if there are multiple heads — merge them with `alembic merge`.

### Streamlit shows "API unreachable"
- In Docker: `API_BASE_URL` should be `http://api:8000`, not `localhost:8000`.
- Locally: `http://localhost:8000`.

### "All my LVs are rejected with `low_presence_rate`"
Either your dataset is too small (resamples don't preserve the pattern) or you're in Production mode (0.70 threshold) on a synthetic dataset. Switch to `"analysis_mode": "demo"`.

### `numpy bool_/int_/float_` JSON serialization error
Already fixed (commit `177d0d8`). If you see it again, the helper at [`src/ive/utils/helpers.py`](src/ive/utils/helpers.py) needs to be invoked before the response is serialized.

### Tests pass locally, fail in CI
CI runs against Postgres 16 + Redis 7 service containers. Local Postgres version differences (e.g., JSONB defaulting differences) can mask issues. Run `make dev && make test-integration` to reproduce.

### Pipeline crashes mid-experiment, status stuck on `running`
Check `experiment_events` — the audit log is independent of the main session and will show the last completed phase. Use `POST /experiments/{id}/cancel` to mark cancelled, or hand-update `status` in DB.

---

## 20. How to Extend IVE

### Add a new detection algorithm
1. Create `src/ive/detection/<your_detector>.py`. Output `Pattern` objects in the standard shape.
2. Wire it into `IVEPipeline._phase_3_detect` in [`src/ive/core/pipeline.py`](src/ive/core/pipeline.py).
3. Add a `pattern_type` literal (e.g., `"changepoint"`) and ensure `PatternScorer` handles it.
4. Add a unit test under `tests/unit/test_<your_detector>.py`.
5. Add a statistical test if you're claiming a new false-positive guarantee.

### Add a new construction rule type
1. Extend `VariableSynthesizer` to emit the new rule shape.
2. Extend `BootstrapValidator` to know how to **re-apply** that rule on a bootstrap resample.
3. Extend `ExplanationGenerator` to produce business-readable prose for it.
4. Document the rule schema in `docs/LLD.md`.

### Add a new ML model
1. Subclass `IVEModel` in `src/ive/models/<your_model>.py`. Implement `fit / predict / get_shap_values`.
2. Add a factory entry the pipeline can dispatch on.
3. Register in `model_capabilities.yaml` if you want UI exposure.
4. Add unit test under `tests/unit/test_<your_model>.py`.

### Add a new API endpoint
1. Add the route to the appropriate file under `src/ive/api/v1/endpoints/`.
2. Define request/response Pydantic models in `src/ive/api/v1/schemas/`.
3. Use `Depends(get_db)` for DB access; never call ML directly — enqueue a Celery task.
4. Add an integration test under `tests/integration/test_api_endpoints.py`.

### Add a new Celery task
1. Define it in `src/ive/workers/tasks.py` with `@celery_app.task(name=...)`.
2. Pick a queue (`default` / `analysis` / `high_priority`) via `task_routes` config.
3. Use `asyncio.run(...)` only at the task boundary; keep the inside async.
4. Add a unit test that uses `CELERY_TASK_ALWAYS_EAGER=True` (set in `tests/conftest.py`).

### Add a database column or table
1. Edit `src/ive/db/models.py`.
2. `make makemigrations MSG="add <thing>"` → review the generated file in `alembic/versions/`.
3. `make migrate` → confirm with `make migration-current`.
4. Update the relevant repository in `src/ive/db/repositories/`.
5. Don't forget integration tests — DDL changes have a habit of surprising prod.

---

## Appendix A — File Inventory Checklist

Use this to verify you've considered every place a change might need to land:

- [ ] `src/ive/main.py` (app factory)
- [ ] `src/ive/config.py` (settings)
- [ ] `src/ive/api/v1/router.py` (route registration)
- [ ] `src/ive/api/v1/endpoints/{datasets,experiments,latent_variables,health}.py`
- [ ] `src/ive/api/v1/schemas/{dataset,experiment,latent_variable}_schemas.py`
- [ ] `src/ive/api/middleware/{auth,error_handler,rate_limit}.py`
- [ ] `src/ive/api/websocket/progress.py`
- [ ] `src/ive/core/pipeline.py`
- [ ] `src/ive/data/{ingestion,profiler,preprocessor,validator}.py`
- [ ] `src/ive/models/{base,linear,xgboost,cross_validator,residual_analyzer}.py`
- [ ] `src/ive/detection/{subgroup_discovery,clustering,shap_interactions,temporal_analysis,pattern_scorer}.py`
- [ ] `src/ive/construction/{variable_synthesizer,bootstrap_validator,causal_checker,explanation_generator}.py`
- [ ] `src/ive/db/{database,models}.py` and `repositories/`
- [ ] `src/ive/storage/artifact_store.py`
- [ ] `src/ive/utils/{logging,statistics,reporting,helpers}.py`
- [ ] `src/ive/workers/{celery_app,tasks}.py`
- [ ] `streamlit_app/app.py` and `pages/01_upload.py` … `05_compare.py`
- [ ] `streamlit_app/components/{sidebar,theme,charts,widgets}.py`
- [ ] `tests/{unit,integration,statistical,e2e,fixtures}/`
- [ ] `alembic/versions/*` (migrations)
- [ ] `docker-compose.yml`, `Dockerfile`, `Makefile`
- [ ] `.env.example` (every new setting must be added here)
- [ ] `.github/workflows/ci.yml`
- [ ] `pyproject.toml` (deps + tool config)
- [ ] `README.md` and `docs/*.md`
- [ ] **This file (`AGENTS.md`)** — keep it in sync.

---

## Appendix B — One-Page Cheat Sheet

```
START          : make dev
LOGS           : make logs / logs-api / logs-worker
TEST           : make test-unit (fast) | make test (full)
LINT           : make lint    FORMAT: make format
MIGRATE        : make migrate
NEW MIGRATION  : make makemigrations MSG="..."
SEED           : make seed
SHELL          : make shell        (python REPL inside API)
STOP           : make down         WIPE: make down-volumes
URLS           : api 8000 · streamlit 8501 · flower 5555 · pg 5432 · redis 6379
AUTH HEADER    : X-API-Key: dev-key-1
SWAGGER        : http://localhost:8000/docs
DEMO DATA      : demo_datasets/ (5 hidden + 1 control)
ANALYSIS MODES : "demo" (loose) | "production" (strict)
PIPELINE       : ingest → CV+SHAP → detect → synthesize+bootstrap → (eval)
GATES          : effect size · BH-adjusted p · bootstrap presence · causal heuristic
QUEUES         : default, analysis, high_priority
EVENT LOG      : experiment_events table (independent psycopg2 transaction)
```

---

## Phase A change log

The Phase A rollout (PRs 1-10) extended IVE with hosted-LLM enrichment, multi-user auth, and a per-column data-egress sensitivity model. This section is the at-a-glance pointer for what changed and where.

| PR | Title | Where |
|---|---|---|
| 1 | LLM package + Phase A migrations | [src/ive/llm/](src/ive/llm/), [src/ive/db/models.py](src/ive/db/models.py), `alembic/versions/20260427_120{0..3}_*.py` |
| 2 | Multi-user auth (api_keys table extension, scopes, audit log) | [src/ive/auth/](src/ive/auth/), [src/ive/api/v1/endpoints/api_keys.py](src/ive/api/v1/endpoints/api_keys.py) |
| 3 | Per-column sensitivity (`dataset_column_metadata` + endpoints + auto-create on upload) | [src/ive/db/repositories/dataset_column_metadata_repo.py](src/ive/db/repositories/dataset_column_metadata_repo.py), [src/ive/api/v1/endpoints/column_metadata.py](src/ive/api/v1/endpoints/column_metadata.py), [src/ive/auth/egress.py](src/ive/auth/egress.py) |
| 4 | `generate_llm_explanations` Celery task + chained from `run_experiment` | [src/ive/workers/llm_enrichment.py](src/ive/workers/llm_enrichment.py), [src/ive/workers/tasks.py](src/ive/workers/tasks.py), [src/ive/db/repositories/llm_explanation_repo.py](src/ive/db/repositories/llm_explanation_repo.py), [src/ive/llm/payloads.py](src/ive/llm/payloads.py), [src/ive/llm/rule_based.py](src/ive/llm/rule_based.py) |
| 5 | `explanation_source` / `llm_explanation_pending` on responses + endpoint LLM-prefer logic | [src/ive/api/v1/schemas/latent_variable_schemas.py](src/ive/api/v1/schemas/latent_variable_schemas.py), [src/ive/api/v1/schemas/experiment_schemas.py](src/ive/api/v1/schemas/experiment_schemas.py), [src/ive/api/v1/endpoints/experiments.py](src/ive/api/v1/endpoints/experiments.py), [src/ive/api/v1/endpoints/latent_variables.py](src/ive/api/v1/endpoints/latent_variables.py) |
| 6 | AI-assisted badge on Streamlit results + dataset-level sensitivity banner | [streamlit_app/components/theme.py](streamlit_app/components/theme.py), [streamlit_app/pages/04_results.py](streamlit_app/pages/04_results.py) |
| 7 | Streamlit Dataset Settings page (column-sensitivity editor) | [streamlit_app/pages/06_dataset_settings.py](streamlit_app/pages/06_dataset_settings.py) |
| 8 | Sensitive-data egress E2E test (CI gate on `docs/RESPONSE_CONTRACT.md` §13) | [tests/integration/test_sensitive_data_egress.py](tests/integration/test_sensitive_data_egress.py) |
| 9 | `docs/RESPONSE_CONTRACT.md` v1 (15 sections, ~3,100 words) | [docs/RESPONSE_CONTRACT.md](docs/RESPONSE_CONTRACT.md) |
| 10 | This update — AGENTS.md threading | (this file) |

**Total Phase A test footprint**: 455 unit + integration tests passing; 4 alembic migrations forward + downgrade clean. Lint clean across all touched files.

---

*Last updated: 2026-04-28 (Phase A complete). When you make architectural or interface changes, update the relevant section AND the [File Inventory Checklist](#appendix-a--file-inventory-checklist).*
