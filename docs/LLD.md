# Low-Level Design — Invisible Variables Engine (IVE)

| Field       | Value                |
| ----------- | -------------------- |
| **Version** | 2.0                  |
| **Authors** | IVE Engineering Team |
| **Date**    | 2026-03-02           |
| **Status**  | Approved             |
| **Parent**  | [HLD v2.0](./HLD.md) |

---

## Table of Contents

1. [Class Diagrams](#1-class-diagrams)
2. [Database Schema](#2-database-schema)
3. [API Contracts](#3-api-contracts)
4. [Algorithm Pseudocode](#4-algorithm-pseudocode)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Error Handling Strategy](#6-error-handling-strategy)
7. [Configuration Schema](#7-configuration-schema)

---

## 1. Class Diagrams

### 1.1 Data Layer

```mermaid
classDiagram
    class DataIngestion {
        -max_rows: int | None
        -supported_formats: list~str~
        +load(path: Path) pd.DataFrame
        +detect_delimiter(path: Path) str
        -_load_csv(path: Path) pd.DataFrame
        -_load_parquet(path: Path) pd.DataFrame
        -_strip_columns(df: pd.DataFrame) pd.DataFrame
    }

    class DataProfiler {
        +profile(df: pd.DataFrame) dict
        +infer_column_types(df: pd.DataFrame) dict~str,str~
        +compute_column_stats(df: pd.DataFrame, col: str) dict
        +detect_multicollinearity(df: pd.DataFrame, threshold: float) list~tuple~
        -_classify_column(series: pd.Series) str
        -_compute_correlation_matrix(df: pd.DataFrame) pd.DataFrame
    }

    class DataValidator {
        -min_rows: int
        -max_missing_pct: float
        +validate(df: pd.DataFrame, target: str) ValidationResult
        +validate_target(df: pd.DataFrame, target: str) bool
        +detect_task_type(y: pd.Series) str
        -_check_schema(df: pd.DataFrame) list~str~
        -_check_missing(df: pd.DataFrame) list~str~
    }

    class DataPreprocessor {
        -scaler_type: str
        -_fitted: bool
        -_pipeline: ColumnTransformer | None
        -_feature_names: list~str~
        +fit_transform(df: pd.DataFrame, features: list~str~) tuple~ndarray, list~str~~
        +transform(df: pd.DataFrame, features: list~str~) ndarray
        -_build_pipeline(df: pd.DataFrame, features: list~str~) ColumnTransformer
    }

    class ValidationResult {
        +is_valid: bool
        +errors: list~str~
        +warnings: list~str~
        +task_type: str
    }

    DataIngestion --> DataProfiler : feeds into
    DataProfiler --> DataValidator : feeds into
    DataValidator --> DataPreprocessor : feeds into
    DataValidator --> ValidationResult : produces
```

### 1.2 Model Layer

```mermaid
classDiagram
    class IVEModel {
        <<abstract>>
        +model_name: str*
        +is_fitted: bool
        +fit(X: ndarray, y: ndarray) None*
        +predict(X: ndarray) ndarray*
        +get_feature_importance() dict~str,float~*
        +get_shap_values(X: ndarray) ndarray*
        +get_params() dict*
    }

    class LinearIVEModel {
        -alpha: float
        -_model: Ridge | None
        -_feature_names: list~str~
        +model_name: str = "linear"
        +fit(X: ndarray, y: ndarray) None
        +predict(X: ndarray) ndarray
        +get_feature_importance() dict~str,float~
        +get_shap_values(X: ndarray) ndarray
        +get_params() dict
    }

    class XGBoostIVEModel {
        -n_estimators: int
        -max_depth: int
        -learning_rate: float
        -_model: XGBRegressor | None
        -_feature_names: list~str~
        +model_name: str = "xgboost"
        +fit(X: ndarray, y: ndarray) None
        +predict(X: ndarray) ndarray
        +get_feature_importance() dict~str,float~
        +get_shap_values(X: ndarray) ndarray
        +get_shap_interaction_values(X: ndarray) ndarray
        +get_params() dict
    }

    class CrossValidator {
        -model: IVEModel
        -n_splits: int
        -random_seed: int
        +fit(X: ndarray, y: ndarray) CVResult
        -_create_folds(X: ndarray, y: ndarray) list~tuple~
    }

    class CVResult {
        +oof_predictions: ndarray
        +oof_residuals: ndarray
        +fold_scores: list~float~
        +fitted_models: list~IVEModel~
        +feature_importances: dict~str,float~
    }

    class ResidualAnalyzer {
        +analyze(residuals: ndarray, X: ndarray) ResidualAnalysis
        +test_normality(residuals: ndarray) tuple~float,float~
        +test_heteroscedasticity(residuals: ndarray, X: ndarray) tuple~float,float~
        +compute_large_residual_pct(residuals: ndarray, threshold: float) float
    }

    class ResidualAnalysis {
        +mean: float
        +std: float
        +skewness: float
        +kurtosis: float
        +normality_stat: float
        +normality_p: float
        +bp_stat: float
        +bp_p: float
        +dw_stat: float
        +large_residual_pct: float
    }

    IVEModel <|-- LinearIVEModel
    IVEModel <|-- XGBoostIVEModel
    CrossValidator o-- IVEModel : uses
    CrossValidator --> CVResult : produces
    ResidualAnalyzer --> ResidualAnalysis : produces
```

### 1.3 Detection Layer

```mermaid
classDiagram
    class SubgroupDiscoverer {
        -beam_width: int
        -min_coverage: float
        -max_depth: int
        -alpha: float
        +discover(df: pd.DataFrame, residuals: ndarray, features: list, top_k: int) list~SubgroupPattern~
        -_generate_selectors(df: pd.DataFrame, feature: str) list~Selector~
        -_refine_subgroup(parent: Selector, df: pd.DataFrame, residuals: ndarray) list~SubgroupPattern~
        -_compute_wracc(mask: ndarray, residuals: ndarray) float
        -_cohens_d(group1: ndarray, group2: ndarray) float
    }

    class SubgroupPattern {
        +rule: str
        +mask: ndarray
        +coverage: float
        +effect_size: float
        +wracc: float
        +p_value: float
        +mean_residual: float
    }

    class HDBSCANClusterer {
        -min_cluster_size: int
        -min_samples: int
        -residual_weight: float
        +fit(residuals: ndarray, X: ndarray) ClusteringResult
        -_build_feature_space(residuals: ndarray, X: ndarray) ndarray
        -_compute_cluster_stats(labels: ndarray, residuals: ndarray) dict
    }

    class ClusteringResult {
        +labels: ndarray
        +n_clusters: int
        +noise_fraction: float
        +cluster_stats: dict~int, dict~
        +probabilities: ndarray
    }

    class SHAPInteractionAnalyzer {
        -sample_size: int
        -seed: int
        +compute(model: IVEModel, X: ndarray, names: list, compute_interactions: bool) SHAPResult
        +rank_interactions(interaction_matrix: ndarray, names: list, top_k: int) list~tuple~
    }

    class SHAPResult {
        +shap_values: ndarray
        +mean_abs_shap: dict~str,float~
        +interaction_matrix: ndarray | None
        +top_interactions: list~tuple~ | None
    }

    class TemporalAnalyzer {
        -min_segment_size: int
        +analyze(residuals: ndarray, timestamps: ndarray) list~TemporalPattern~
        +detect_trend(residuals: ndarray) dict
        +detect_seasonality(residuals: ndarray) dict
        +detect_changepoints(residuals: ndarray) list~int~
    }

    class PatternScorer {
        -effect_weight: float
        -stability_weight: float
        -coverage_weight: float
        -min_score: float
        +score_and_rank(patterns: list, residuals: ndarray) list~ScoredPattern~
        +deduplicate(patterns: list, jaccard_threshold: float) list~ScoredPattern~
        -_composite_score(effect: float, stability: float, coverage: float) float
    }

    class ScoredPattern {
        +pattern: SubgroupPattern | ClusteringResult
        +score: float
        +rank: int
        +source: str
    }

    SubgroupDiscoverer --> SubgroupPattern
    HDBSCANClusterer --> ClusteringResult
    SHAPInteractionAnalyzer --> SHAPResult
    PatternScorer --> ScoredPattern
```

### 1.4 Construction Layer

```mermaid
classDiagram
    class VariableSynthesizer {
        +synthesize(patterns: list, shap_result: SHAPResult, df: pd.DataFrame, features: list) list~LatentVariableCandidate~
        -_pattern_to_candidate(pattern: ScoredPattern, df: pd.DataFrame, features: list) LatentVariableCandidate
        -_generate_candidate_name(features: list) str | None
        -_compute_feature_correlations(mask: ndarray, df: pd.DataFrame, features: list) dict
    }

    class LatentVariableCandidate {
        +rank: int
        +name: str | None
        +description: str | None
        +confidence_score: float
        +effect_size: float
        +coverage_pct: float
        +candidate_features: list~str~
        +construction_rule: dict | None
        +validation_results: dict | None
    }

    class BootstrapValidator {
        -n_iterations: int
        -confidence_level: float
        -seed: int
        +validate(candidate: LatentVariableCandidate, X: ndarray, y: ndarray) BootstrapResult
        -_single_bootstrap(candidate: LatentVariableCandidate, X: ndarray, y: ndarray, rng: Generator) float
    }

    class BootstrapResult {
        +stability_score: float
        +mean_effect_size: float
        +ci_lower: float
        +ci_upper: float
        +p_value: float
        +n_iterations: int
    }

    class CausalChecker {
        -correlation_threshold: float
        +check(candidate: LatentVariableCandidate, df: pd.DataFrame, target: str) CausalResult
        +check_reverse_causality(candidate: LatentVariableCandidate, df: pd.DataFrame, target: str) bool
        +check_confounding_proxy(candidate: LatentVariableCandidate, df: pd.DataFrame, features: list) bool
    }

    class ExplanationGenerator {
        +generate(candidate: LatentVariableCandidate, profile: dict) str
        +generate_batch(candidates: list, profile: dict) list~str~
        -_select_template(candidate: LatentVariableCandidate) str
        -_fill_template(template: str, candidate: LatentVariableCandidate, profile: dict) str
    }

    VariableSynthesizer --> LatentVariableCandidate
    BootstrapValidator --> BootstrapResult
    CausalChecker --> "CausalResult"
```

### 1.5 Orchestration & Infrastructure

```mermaid
classDiagram
    class IVEEngine {
        -config: ExperimentConfig
        +run(experiment_id: UUID, config: ExperimentConfig, data_path: str) EngineResult
        -_execute_phase(phase: PhaseBase, ctx: PipelineContext) PhaseResult
        -_publish_progress(experiment_id: UUID, phase: str, pct: int) None
    }

    class PipelineContext {
        +experiment_id: UUID
        +config: ExperimentConfig
        +df: pd.DataFrame | None
        +X: ndarray | None
        +y: ndarray | None
        +feature_names: list~str~
        +column_types: dict~str,str~
        +residuals: dict~str,ndarray~
        +fitted_models: dict~str,list~IVEModel~~
        +patterns: list~ScoredPattern~
        +latent_variables: list~LatentVariableCandidate~
    }

    class PhaseBase {
        <<abstract>>
        +name: str*
        +execute(ctx: PipelineContext) PhaseResult*
    }

    class ArtifactStore {
        <<abstract>>
        +save_bytes(key: str, data: bytes) str*
        +load_bytes(key: str) bytes*
        +exists(key: str) bool*
        +delete(key: str) bool*
        +save_json(key: str, data: Any) str
        +load_json(key: str) Any
        +save_numpy(key: str, array: ndarray) str
        +save_pickle(key: str, obj: Any) str
    }

    class LocalArtifactStore {
        -base_dir: Path
        +save_bytes(key: str, data: bytes) str
        +load_bytes(key: str) bytes
        +exists(key: str) bool
        +delete(key: str) bool
    }

    class S3ArtifactStore {
        -bucket: str
        -region: str
        +save_bytes(key: str, data: bytes) str
        +load_bytes(key: str) bytes
        +exists(key: str) bool
        +delete(key: str) bool
    }

    PhaseBase <|-- PhaseUnderstand
    PhaseBase <|-- PhaseModel
    PhaseBase <|-- PhaseDetect
    PhaseBase <|-- PhaseConstruct
    IVEEngine o-- PhaseBase
    IVEEngine o-- PipelineContext
    ArtifactStore <|-- LocalArtifactStore
    ArtifactStore <|-- S3ArtifactStore
```

---

## 2. Database Schema

### 2.1 Enum Types

```sql
CREATE TYPE experiment_status AS ENUM (
    'queued', 'running', 'completed', 'failed', 'cancelled'
);

CREATE TYPE model_type AS ENUM (
    'linear', 'logistic', 'xgboost'
);

CREATE TYPE pattern_type AS ENUM (
    'subgroup', 'cluster', 'interaction', 'temporal'
);

CREATE TYPE latent_variable_status AS ENUM (
    'candidate', 'validated', 'rejected'
);
```

### 2.2 Table: `datasets`

```sql
CREATE TABLE datasets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    file_path       TEXT NOT NULL,
    row_count       INTEGER NOT NULL DEFAULT 0,
    column_count    INTEGER NOT NULL DEFAULT 0,
    target_column   VARCHAR(255) NOT NULL,
    time_column     VARCHAR(255),
    checksum        VARCHAR(64),           -- SHA-256 hex digest
    profile_path    TEXT,                   -- path to profile JSON artifact
    schema_json     JSONB NOT NULL DEFAULT '{}',
    status          VARCHAR(50) NOT NULL DEFAULT 'uploaded',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_created ON datasets(created_at DESC);
```

**`schema_json` JSONB structure:**

```json
{
  "columns": {
    "age": {
      "dtype": "float64",
      "inferred_type": "continuous",
      "nullable": false,
      "unique_count": 45
    },
    "region": {
      "dtype": "object",
      "inferred_type": "categorical",
      "nullable": false,
      "unique_count": 4
    }
  },
  "missing_summary": { "age": 0.0, "region": 0.02 },
  "correlation_pairs": [["age", "income", 0.72]]
}
```

### 2.3 Table: `experiments`

```sql
CREATE TABLE experiments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id      UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    config_json     JSONB NOT NULL DEFAULT '{}',
    status          experiment_status NOT NULL DEFAULT 'queued',
    progress_pct    INTEGER NOT NULL DEFAULT 0,
    current_stage   VARCHAR(50),
    task_id         VARCHAR(255),          -- Celery task ID
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_experiments_dataset ON experiments(dataset_id);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created ON experiments(created_at DESC);
```

**`config_json` JSONB structure:**

```json
{
  "target_column": "price",
  "model_types": ["linear", "xgboost"],
  "cv_folds": 5,
  "min_cluster_size": 10,
  "shap_sample_size": 500,
  "max_latent_variables": 5,
  "random_seed": 42,
  "alpha": 0.05,
  "min_subgroup_coverage": 0.03,
  "bootstrap_iterations": 1000
}
```

### 2.4 Table: `models`

```sql
CREATE TABLE models (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id           UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_type              model_type NOT NULL,
    fold_number             INTEGER NOT NULL,
    train_metric            DOUBLE PRECISION,
    val_metric              DOUBLE PRECISION,
    artifact_path           TEXT,
    hyperparams             JSONB NOT NULL DEFAULT '{}',
    feature_importances     JSONB NOT NULL DEFAULT '{}',
    training_time_seconds   DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (experiment_id, model_type, fold_number)
);

CREATE INDEX idx_models_experiment ON models(experiment_id);
```

**`feature_importances` JSONB:**

```json
{ "age": 0.35, "income": 0.28, "region_north": 0.12, "education_master": 0.08 }
```

### 2.5 Table: `residuals`

> **Design note:** For datasets with >100K rows, individual residuals are stored as NumPy artifacts via `ArtifactStore` instead of in PostgreSQL. This table stores aggregate per-fold summaries.

```sql
CREATE TABLE residuals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_type      model_type NOT NULL,
    fold_number     INTEGER NOT NULL,
    sample_count    INTEGER NOT NULL,
    mean_residual   DOUBLE PRECISION NOT NULL,
    std_residual    DOUBLE PRECISION NOT NULL,
    skewness        DOUBLE PRECISION,
    kurtosis        DOUBLE PRECISION,
    artifact_path   TEXT NOT NULL,         -- path to full residual ndarray
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (experiment_id, model_type, fold_number)
);

CREATE INDEX idx_residuals_experiment ON residuals(experiment_id);
```

### 2.6 Table: `error_patterns`

```sql
CREATE TABLE error_patterns (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id       UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    pattern_type        pattern_type NOT NULL,
    rank                INTEGER NOT NULL,
    subgroup_definition JSONB,
    effect_size         DOUBLE PRECISION NOT NULL,
    p_value             DOUBLE PRECISION,
    adjusted_p_value    DOUBLE PRECISION,
    sample_count        INTEGER NOT NULL,
    coverage_pct        DOUBLE PRECISION NOT NULL,
    stability_score     DOUBLE PRECISION,
    mean_residual       DOUBLE PRECISION NOT NULL,
    std_residual        DOUBLE PRECISION NOT NULL,
    composite_score     DOUBLE PRECISION NOT NULL,
    evidence            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_patterns_experiment ON error_patterns(experiment_id);
CREATE INDEX idx_patterns_type ON error_patterns(pattern_type);
```

**`subgroup_definition` JSONB:**

```json
{
  "rule": "age > 45 AND region IN ('north', 'south')",
  "features": ["age", "region"],
  "selectors": [
    { "feature": "age", "operator": ">", "value": 45 },
    { "feature": "region", "operator": "in", "value": ["north", "south"] }
  ]
}
```

**`evidence` JSONB:**

```json
{
  "ks_statistic": 0.34,
  "ks_p_value": 0.0001,
  "cohens_d": 0.82,
  "mean_abs_residual_inside": 4.2,
  "mean_abs_residual_outside": 1.1,
  "shap_features": { "age": 0.45, "income": 0.32 }
}
```

### 2.7 Table: `latent_variables`

```sql
CREATE TABLE latent_variables (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id               UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    rank                        INTEGER NOT NULL,
    name                        VARCHAR(255),
    description                 TEXT,
    construction_rule           JSONB,
    source_pattern_ids          UUID[] NOT NULL DEFAULT '{}',
    candidate_features          JSONB NOT NULL DEFAULT '[]',
    importance_score            DOUBLE PRECISION NOT NULL,
    stability_score             DOUBLE PRECISION NOT NULL,
    effect_size                 DOUBLE PRECISION NOT NULL,
    coverage_pct                DOUBLE PRECISION NOT NULL,
    bootstrap_presence_rate     DOUBLE PRECISION,
    model_improvement_pct       DOUBLE PRECISION,
    confidence_interval_lower   DOUBLE PRECISION,
    confidence_interval_upper   DOUBLE PRECISION,
    explanation_text            TEXT,
    status                      latent_variable_status NOT NULL DEFAULT 'candidate',
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lv_experiment ON latent_variables(experiment_id);
CREATE INDEX idx_lv_status ON latent_variables(status);
```

### 2.8 Table: `api_keys`

```sql
CREATE TABLE api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash    VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of the raw key
    name        VARCHAR(255) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '["read", "write"]',
    rate_limit  INTEGER NOT NULL DEFAULT 100, -- requests per minute
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;
```

### 2.9 Table: `experiment_events`

```sql
CREATE TABLE experiment_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase           VARCHAR(50),
    event_type      VARCHAR(100) NOT NULL,
    payload         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_experiment ON experiment_events(experiment_id);
CREATE INDEX idx_events_created ON experiment_events(created_at DESC);
```

---

## 3. API Contracts

### 3.0 Common Conventions

**Base URL:** `http://localhost:8000/api/v1`

**Required Headers (all protected endpoints):**

| Header         | Value                                                     | Required |
| -------------- | --------------------------------------------------------- | -------- |
| `X-API-Key`    | Valid API key string                                      | Yes      |
| `Content-Type` | `application/json` (or `multipart/form-data` for uploads) | Yes      |

**Standard Error Response (all endpoints):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable error description",
    "details": [{ "field": "name", "message": "Field is required" }]
  }
}
```

| HTTP Status | Code               | Meaning                              |
| ----------- | ------------------ | ------------------------------------ |
| 400         | `BAD_REQUEST`      | Malformed request body               |
| 401         | `UNAUTHORIZED`     | Missing or invalid API key           |
| 404         | `NOT_FOUND`        | Resource does not exist              |
| 422         | `VALIDATION_ERROR` | Request body fails schema validation |
| 429         | `RATE_LIMITED`     | Too many requests                    |
| 500         | `INTERNAL_ERROR`   | Unexpected server error              |

**Pagination (all list endpoints):**

Query params: `?offset=0&limit=20`

Response wrapper:

```json
{
  "items": [...],
  "total": 150,
  "offset": 0,
  "limit": 20
}
```

---

### 3.1 POST /api/v1/datasets

Upload a dataset file with metadata.

**Request:** `Content-Type: multipart/form-data`

| Field           | Type                 | Required | Validation                               |
| --------------- | -------------------- | -------- | ---------------------------------------- |
| `file`          | Binary (CSV/Parquet) | Yes      | Max 500 MB, extensions `.csv` `.parquet` |
| `name`          | string               | Yes      | 1–255 chars                              |
| `target_column` | string               | Yes      | 1–255 chars                              |
| `description`   | string               | No       | Max 2000 chars                           |
| `time_column`   | string               | No       | Column name if temporal analysis desired |

**Response 201:**

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Housing Prices",
  "target_column": "price",
  "status": "uploaded",
  "row_count": 0,
  "column_count": 0,
  "created_at": "2026-03-02T10:00:00Z"
}
```

**Example curl:**

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H "X-API-Key: my-api-key" \
  -F "file=@housing.csv" \
  -F "name=Housing Prices" \
  -F "target_column=price"
```

---

### 3.2 GET /api/v1/datasets

**Query params:** `offset` (int, default 0), `limit` (int, default 20), `search` (string, optional — name ilike filter)

**Response 200:**

```json
{
  "items": [
    {
      "id": "a1b2c3d4-...",
      "name": "Housing Prices",
      "target_column": "price",
      "row_count": 1000,
      "column_count": 10,
      "status": "profiled",
      "created_at": "2026-03-02T10:00:00Z"
    }
  ],
  "total": 1,
  "offset": 0,
  "limit": 20
}
```

---

### 3.3 GET /api/v1/datasets/{id}

**Response 200:**

```json
{
  "id": "a1b2c3d4-...",
  "name": "Housing Prices",
  "description": "Q1 2024 housing dataset",
  "target_column": "price",
  "time_column": null,
  "row_count": 1000,
  "column_count": 10,
  "checksum": "sha256:abc123...",
  "status": "profiled",
  "schema": {
    "columns": {
      "age": {
        "dtype": "float64",
        "inferred_type": "continuous",
        "nullable": false,
        "unique_count": 45
      }
    },
    "missing_summary": { "age": 0.0 }
  },
  "created_at": "2026-03-02T10:00:00Z",
  "updated_at": "2026-03-02T10:01:00Z"
}
```

---

### 3.4 DELETE /api/v1/datasets/{id}

**Response 204:** No body. Cascades to all experiments.

**Response 404:** Dataset not found.

---

### 3.5 POST /api/v1/experiments

**Request body:**

```json
{
  "dataset_id": "a1b2c3d4-...",
  "name": "Housing Experiment v1",
  "config": {
    "model_types": ["linear", "xgboost"],
    "cv_folds": 5,
    "min_cluster_size": 10,
    "shap_sample_size": 500,
    "max_latent_variables": 5,
    "random_seed": 42,
    "alpha": 0.05,
    "min_subgroup_coverage": 0.03,
    "bootstrap_iterations": 1000
  }
}
```

| Field                          | Type     | Required | Validation                                                    |
| ------------------------------ | -------- | -------- | ------------------------------------------------------------- |
| `dataset_id`                   | UUID     | Yes      | Must exist with status ≠ "uploaded"                           |
| `name`                         | string   | Yes      | 1–255 chars                                                   |
| `config.model_types`           | string[] | No       | Default `["linear", "xgboost"]`. Allowed: `linear`, `xgboost` |
| `config.cv_folds`              | int      | No       | Default 5. Range: 2–10                                        |
| `config.min_cluster_size`      | int      | No       | Default 10. Range: 5–100                                      |
| `config.shap_sample_size`      | int      | No       | Default 500. Range: 50–5000                                   |
| `config.max_latent_variables`  | int      | No       | Default 5. Range: 1–20                                        |
| `config.random_seed`           | int      | No       | Default 42                                                    |
| `config.alpha`                 | float    | No       | Default 0.05. Range: 0.001–0.1                                |
| `config.min_subgroup_coverage` | float    | No       | Default 0.03. Range: 0.01–0.5                                 |
| `config.bootstrap_iterations`  | int      | No       | Default 1000. Range: 100–10000                                |

**Response 201:**

```json
{
  "id": "b2c3d4e5-...",
  "dataset_id": "a1b2c3d4-...",
  "name": "Housing Experiment v1",
  "status": "queued",
  "task_id": "celery-task-uuid",
  "created_at": "2026-03-02T10:05:00Z"
}
```

---

### 3.6 GET /api/v1/experiments

**Query params:** `offset`, `limit`, `dataset_id` (UUID, optional), `status` (string, optional)

**Response 200:** Paginated list of experiment summaries (same schema as 3.5 response plus `progress_pct`, `current_stage`).

---

### 3.7 GET /api/v1/experiments/{id}

**Response 200:**

```json
{
  "id": "b2c3d4e5-...",
  "dataset_id": "a1b2c3d4-...",
  "name": "Housing Experiment v1",
  "config": { "...full config_json..." },
  "status": "completed",
  "progress_pct": 100,
  "current_stage": "construct",
  "error_message": null,
  "created_at": "2026-03-02T10:05:00Z",
  "started_at": "2026-03-02T10:05:01Z",
  "completed_at": "2026-03-02T10:07:30Z",
  "n_latent_variables": 3,
  "n_patterns": 8
}
```

---

### 3.8 POST /api/v1/experiments/{id}/cancel

**Response 200:**

```json
{ "id": "b2c3d4e5-...", "status": "cancelled" }
```

**Response 409:** Experiment already completed/failed/cancelled.

---

### 3.9 GET /api/v1/experiments/{id}/residuals

**Query params:** `model_type` (optional), `offset`, `limit`

**Response 200:**

```json
{
  "items": [
    {
      "model_type": "xgboost",
      "fold_number": 0,
      "sample_count": 200,
      "mean_residual": -0.02,
      "std_residual": 1.45,
      "skewness": 0.12,
      "kurtosis": 3.05
    }
  ],
  "total": 10
}
```

---

### 3.10 GET /api/v1/experiments/{id}/patterns

**Response 200:**

```json
{
  "items": [
    {
      "id": "c3d4e5f6-...",
      "pattern_type": "subgroup",
      "rank": 1,
      "rule": "age > 45 AND region IN ('north')",
      "effect_size": 0.82,
      "p_value": 0.0001,
      "sample_count": 120,
      "coverage_pct": 12.0,
      "composite_score": 0.78,
      "evidence": { "cohens_d": 0.82, "ks_statistic": 0.34 }
    }
  ],
  "total": 8
}
```

---

### 3.11 GET /api/v1/experiments/{id}/latent-variables

**Response 200:**

```json
{
  "items": [
    {
      "id": "d4e5f6a7-...",
      "rank": 1,
      "name": "Neighbourhood Quality Factor",
      "description": "Hidden variable proxied by zip_code and commute_mins",
      "candidate_features": ["zip_code", "commute_mins"],
      "importance_score": 0.87,
      "stability_score": 0.91,
      "effect_size": 0.82,
      "coverage_pct": 23.5,
      "bootstrap_presence_rate": 0.94,
      "confidence_interval_lower": 0.65,
      "confidence_interval_upper": 0.99,
      "explanation_text": "The model consistently under-estimates...",
      "status": "validated"
    }
  ],
  "total": 3
}
```

---

### 3.12 GET /api/v1/experiments/{id}/report

**Response 200:** `Content-Type: application/json` (full JSON report).

Alternatively returns PDF: `Accept: application/pdf` (future).

---

### 3.13 WS /api/v1/ws/experiments/{id}/progress

**WebSocket protocol:**

1. Client connects: `ws://localhost:8000/api/v1/ws/experiments/{id}/progress`
2. Server sends JSON frames:

```json
{
  "phase": "model",
  "progress_pct": 45,
  "message": "Training XGBoost fold 3/5",
  "timestamp": "2026-03-02T10:06:15Z"
}
```

3. Final frame: `{"phase": "complete", "progress_pct": 100, "message": "Experiment completed", "status": "completed"}`
4. Server closes connection after final frame.

---

### 3.14 GET /api/v1/health

**No authentication required.**

**Response 200:**

```json
{ "status": "ok", "version": "0.1.0", "timestamp": "2026-03-02T10:00:00Z" }
```

---

### 3.15 GET /api/v1/health/ready

**No authentication required.**

**Response 200 (all healthy):**

```json
{
  "status": "ok",
  "checks": {
    "database": { "status": "ok", "latency_ms": 2 },
    "redis": { "status": "ok", "latency_ms": 1 }
  }
}
```

**Response 503 (degraded):**

```json
{
  "status": "degraded",
  "checks": {
    "database": { "status": "ok", "latency_ms": 2 },
    "redis": { "status": "error", "message": "Connection refused" }
  }
}
```

---

## 4. Algorithm Pseudocode

### 4a. Out-of-Fold Residual Generation

```
FUNCTION generate_oof_residuals(D, y, model_types, k_folds, seed):
    INPUT:
        D           — DataFrame of shape (n_samples, n_features)
        y           — target array of shape (n_samples,)
        model_types — list of strings, e.g. ["linear", "xgboost"]
        k_folds     — integer, number of CV folds (default 5)
        seed        — integer, random seed for reproducibility

    OUTPUT:
        residuals   — dict[model_type → ndarray of shape (n_samples,)]
        models      — dict[model_type → list of k fitted model objects]

    1. X ← preprocess(D)                           # encode categoricals, scale numerics
    2. feature_names ← column names after preprocessing
    3. folds ← KFold(n_splits=k_folds, shuffle=True, random_state=seed).split(X, y)

    4. FOR EACH model_type IN model_types:
        a. oof_predictions ← zeros(n_samples)
        b. fold_models ← []
        c. fold_scores ← []

        d. FOR fold_idx, (train_idx, val_idx) IN enumerate(folds):
            i.   X_train, X_val ← X[train_idx], X[val_idx]
            ii.  y_train, y_val ← y[train_idx], y[val_idx]
            iii. model ← deep_copy(create_model(model_type))
            iv.  model.fit(X_train, y_train)
            v.   preds ← model.predict(X_val)
            vi.  oof_predictions[val_idx] ← preds
            vii. score ← r2_score(y_val, preds)
            viii.fold_models.append(model)
            ix.  fold_scores.append(score)
            x.   PUBLISH progress(fold_idx + 1, k_folds)

        e. residuals[model_type] ← y - oof_predictions
        f. models[model_type] ← fold_models

    5. RETURN residuals, models
```

**Invariant:** Every sample appears in exactly one validation fold, so `oof_predictions[i]` is always the prediction made by a model that _never saw_ sample `i` during training. This makes residuals unbiased.

---

### 4b. Subgroup Discovery (Beam Search)

```
FUNCTION discover_subgroups(df, residuals, features, alpha, min_coverage, beam_width, max_depth):
    INPUT:
        df           — original DataFrame
        residuals    — ndarray of shape (n_samples,)
        features     — list of feature names to explore
        alpha        — significance level (default 0.05)
        min_coverage — minimum fraction of samples (default 0.03)
        beam_width   — number of candidates to keep per level (default 10)
        max_depth    — maximum conjunction depth (default 3)

    OUTPUT:
        patterns — list of SubgroupPattern, sorted by WRAcc descending

    1. global_mean ← mean(residuals)
    2. global_std  ← std(residuals)
    3. n ← len(residuals)
    4. min_samples ← n × min_coverage

    5. # Generate initial selectors (depth = 1)
       candidates ← []
       FOR EACH feature IN features:
           IF type(feature) == continuous:
               # Discretise into quartile bins
               thresholds ← percentile(df[feature], [25, 50, 75])
               FOR EACH t IN thresholds:
                   mask_le ← df[feature] ≤ t
                   mask_gt ← df[feature] > t
                   candidates.append(Selector(f"{feature} ≤ {t}", mask_le))
                   candidates.append(Selector(f"{feature} > {t}", mask_gt))
           ELSE IF type(feature) == categorical:
               FOR EACH value IN df[feature].unique():
                   mask_eq ← df[feature] == value
                   candidates.append(Selector(f"{feature} = {value}", mask_eq))

    6. # Evaluate and rank candidates
       scored ← []
       FOR EACH c IN candidates:
           IF c.mask.sum() < min_samples:
               CONTINUE
           subgroup_residuals ← residuals[c.mask]
           complement_residuals ← residuals[~c.mask]
           coverage ← c.mask.sum() / n
           effect ← cohens_d(subgroup_residuals, complement_residuals)
           ks_stat, ks_p ← ks_test(subgroup_residuals, complement_residuals)
           wracc ← coverage × (mean(subgroup_residuals) - global_mean)
           scored.append(SubgroupPattern(c.rule, c.mask, coverage, effect, wracc, ks_p))

    7. # Bonferroni correction
       n_tests ← len(scored)
       FOR EACH s IN scored:
           s.adjusted_p ← min(s.p_value × n_tests, 1.0)

    8. # Filter significant + beam prune
       scored ← [s for s IN scored IF s.adjusted_p < alpha AND |s.effect| > 0.2]
       scored ← sorted(scored, key=wracc, reverse=True)[:beam_width]

    9. # Refinement: depth 2..max_depth
       FOR depth IN 2..max_depth:
           refined ← []
           FOR EACH parent IN scored:
               FOR EACH child_selector IN candidates:
                   combined_mask ← parent.mask AND child_selector.mask
                   IF combined_mask.sum() < min_samples:
                       CONTINUE
                   IF jaccard(parent.mask, combined_mask) > 0.95:
                       CONTINUE   # no new information
                   # Re-evaluate combined subgroup
                   ... (same scoring as step 6) ...
                   refined.append(combined_pattern)
           scored ← top_k(scored + refined, beam_width, by=wracc)

   10. RETURN scored
```

---

### 4c. HDBSCAN Clustering on Residual Space

```
FUNCTION cluster_residuals(residuals, X, min_cluster_size, min_samples, residual_weight):
    INPUT:
        residuals       — ndarray of shape (n_samples,)
        X               — feature matrix of shape (n_samples, n_features)
        min_cluster_size — minimum points per cluster (default 10)
        min_samples      — core point threshold (default 5)
        residual_weight  — up-weight residuals in joint space (default 2.0)

    OUTPUT:
        ClusteringResult with labels, stats, noise fraction

    1. # Build joint feature space
       X_scaled ← StandardScaler().fit_transform(X)
       r_scaled ← StandardScaler().fit_transform(residuals.reshape(-1, 1)) × residual_weight
       Z ← hstack([X_scaled, r_scaled])                     # shape (n, n_features + 1)

    2. # Fit HDBSCAN
       clusterer ← HDBSCAN(
           min_cluster_size=min_cluster_size,
           min_samples=min_samples,
           core_dist_n_jobs=-1        # parallel distance computation
       )
       labels ← clusterer.fit_predict(Z)
       probabilities ← clusterer.probabilities_

    3. # Characterise each cluster
       unique_labels ← set(labels) - {-1}
       n_clusters ← len(unique_labels)
       noise_fraction ← (labels == -1).sum() / len(labels)
       cluster_stats ← {}

       FOR EACH cluster_id IN unique_labels:
           mask ← labels == cluster_id
           cluster_residuals ← residuals[mask]
           complement_residuals ← residuals[~mask]
           stats ← {
               "n_samples": mask.sum(),
               "mean_residual": mean(cluster_residuals),
               "std_residual": std(cluster_residuals),
               "effect_size": cohens_d(cluster_residuals, complement_residuals),
               "mean_probability": mean(probabilities[mask])
           }
           # Statistical test: are cluster residuals different from rest?
           _, p_value ← ks_test(cluster_residuals, complement_residuals)
           stats["p_value"] ← p_value
           cluster_stats[cluster_id] ← stats

    4. RETURN ClusteringResult(labels, n_clusters, noise_fraction, cluster_stats, probabilities)
```

---

### 4d. SHAP Interaction Detection

```
FUNCTION detect_shap_interactions(model, X, feature_names, sample_size, top_k):
    INPUT:
        model         — fitted XGBoostIVEModel
        X             — feature matrix of shape (n_samples, n_features)
        feature_names — list of feature name strings
        sample_size   — max rows for SHAP computation (default 500)
        top_k         — number of top interactions to return (default 10)

    OUTPUT:
        SHAPResult with shap_values, mean_abs_shap, interaction_matrix, top_interactions

    1. # Subsample for performance
       IF len(X) > sample_size:
           indices ← random_sample(range(len(X)), sample_size, seed=42)
           X_sample ← X[indices]
       ELSE:
           X_sample ← X

    2. # Compute SHAP values
       explainer ← shap.TreeExplainer(model._model)
       shap_values ← explainer.shap_values(X_sample)            # shape (sample, features)

    3. # Global feature importance (mean |SHAP|)
       mean_abs_shap ← {}
       FOR i, name IN enumerate(feature_names):
           mean_abs_shap[name] ← mean(abs(shap_values[:, i]))

    4. # Compute interaction values (O(n × d²), expensive)
       interaction_values ← explainer.shap_interaction_values(X_sample)
                           # shape (sample, features, features)

    5. # Average absolute interaction matrix
       mean_interaction ← mean(abs(interaction_values), axis=0)  # shape (d, d)
       # Zero out diagonal (main effects)
       fill_diagonal(mean_interaction, 0)

    6. # Rank interaction pairs
       pairs ← []
       FOR i IN range(n_features):
           FOR j IN range(i+1, n_features):
               pairs.append((feature_names[i], feature_names[j], mean_interaction[i, j]))
       pairs ← sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]

    7. RETURN SHAPResult(shap_values, mean_abs_shap, mean_interaction, pairs)
```

---

### 4e. Bootstrap Stability Validation

```
FUNCTION bootstrap_validate(candidate, X, y, residuals, n_iterations, confidence_level, seed):
    INPUT:
        candidate       — LatentVariableCandidate (has .candidate_features, .effect_size)
        X               — feature matrix (n, d)
        y               — target array (n,)
        residuals       — OOF residual array (n,)
        n_iterations    — number of bootstrap resamples (default 1000)
        confidence_level — CI level (default 0.95)
        seed            — random seed

    OUTPUT:
        BootstrapResult with stability_score, mean_effect, CI, p_value

    1. rng ← RandomGenerator(seed)
    2. n ← len(X)
    3. observed_effect ← candidate.effect_size
    4. boot_effects ← []
    5. n_detected ← 0              # count of resamples where pattern is re-discovered

    6. FOR b IN 1..n_iterations:
        a. indices ← rng.choice(range(n), size=n, replace=True)
        b. X_boot ← X[indices]
        c. y_boot ← y[indices]
        d. r_boot ← residuals[indices]

        e. # Re-apply detection on bootstrap sample
           # (Simplified: check if feature-based subgroup still shows effect)
           mask_boot ← apply_construction_rule(candidate.construction_rule, X_boot)
           IF mask_boot.sum() < 5:
               CONTINUE   # too few samples in this resample

        f. inside  ← r_boot[mask_boot]
           outside ← r_boot[~mask_boot]
           effect  ← cohens_d(inside, outside)
           boot_effects.append(effect)

        g. IF |effect| > 0.2:        # re-detected threshold
               n_detected += 1

    7. boot_effects ← array(boot_effects)
    8. alpha ← 1 - confidence_level
    9. ci_lower ← percentile(boot_effects, 100 × alpha / 2)
   10. ci_upper ← percentile(boot_effects, 100 × (1 - alpha / 2))
   11. stability_score ← n_detected / n_iterations
   12. mean_effect ← mean(boot_effects)

   13. # Permutation p-value: fraction of bootstrap effects ≥ |observed|
       p_value ← mean(abs(boot_effects) >= abs(observed_effect))

   14. RETURN BootstrapResult(stability_score, mean_effect, ci_lower, ci_upper, p_value, n_iterations)
```

**Decision rule:** A candidate is `validated` if `stability_score ≥ 0.70` AND `ci_lower > 0`. Otherwise `rejected`.

---

### 4f. Latent Variable Synthesis

```
FUNCTION synthesize_latent_variables(scored_patterns, shap_result, df, features, max_lvs):
    INPUT:
        scored_patterns — list of ScoredPattern, ranked by composite score
        shap_result     — SHAPResult from SHAP analysis
        df              — original DataFrame
        features        — feature names
        max_lvs         — maximum number of LVs to return

    OUTPUT:
        candidates — list of LatentVariableCandidate, ranked

    1. candidates ← []

    2. FOR rank, pattern IN enumerate(scored_patterns, start=1):
        IF rank > max_lvs:
            BREAK

        a. # Extract sample mask from pattern
           mask ← pattern.mask

        b. # Identify correlated features
           correlations ← {}
           FOR EACH feature IN features:
               IF type(df[feature]) == numeric:
                   corr, p ← pointbiserialr(mask.astype(int), df[feature])
                   IF p < 0.05:
                       correlations[feature] ← abs(corr)
           top_features ← sorted(correlations, key=correlations.get, reverse=True)[:5]

        c. # Cross-reference with SHAP importance
           IF shap_result IS NOT None:
               shap_overlap ← [f for f IN top_features IF shap_result.mean_abs_shap.get(f, 0) > median(shap_result.mean_abs_shap.values())]
               top_features ← shap_overlap OR top_features   # prefer SHAP-validated

        d. # Generate name
           IF len(top_features) >= 2:
               name ← f"Latent {top_features[0].title()} / {top_features[1].title()} Factor"
           ELSE IF len(top_features) == 1:
               name ← f"Latent {top_features[0].title()} Factor"
           ELSE:
               name ← f"Unnamed Latent Variable #{rank}"

        e. # Build construction rule
           construction_rule ← serialise_pattern_to_rule(pattern)

        f. candidate ← LatentVariableCandidate(
               rank=rank,
               name=name,
               confidence_score=pattern.score,
               effect_size=pattern.effect_size,
               coverage_pct=pattern.coverage × 100,
               candidate_features=top_features,
               construction_rule=construction_rule
           )
           candidates.append(candidate)

    3. # Deduplicate by Jaccard overlap
       final ← []
       FOR EACH c IN candidates:
           mask_c ← apply_rule(c.construction_rule, df)
           is_duplicate ← False
           FOR EACH existing IN final:
               mask_e ← apply_rule(existing.construction_rule, df)
               IF jaccard(mask_c, mask_e) > 0.90:
                   is_duplicate ← True
                   BREAK
           IF NOT is_duplicate:
               final.append(c)

    4. # Re-rank final list
       FOR i, c IN enumerate(final, start=1):
           c.rank ← i

    5. RETURN final
```

---

## 5. Sequence Diagrams

### 5.1 Complete Experiment Lifecycle

```mermaid
sequenceDiagram
    actor U as User
    participant ST as Streamlit
    participant API as FastAPI
    participant DB as PostgreSQL
    participant R as Redis
    participant W as Celery Worker
    participant AS as Artifact Store

    U->>ST: Upload CSV file
    ST->>API: POST /api/v1/datasets
    API->>AS: Save file to artifacts/datasets/{id}/raw.csv
    API->>DB: INSERT datasets (status=uploaded)
    API->>R: Enqueue profile_dataset(dataset_id)
    API-->>ST: 201 {dataset_id}
    ST-->>U: "Dataset uploaded!"

    R->>W: Deliver profile_dataset
    W->>AS: Load raw file
    W->>W: Run DataProfiler.profile()
    W->>AS: Save profile.json
    W->>DB: UPDATE datasets (status=profiled, row_count, col_count, schema_json)

    U->>ST: Configure experiment
    ST->>API: POST /api/v1/experiments {dataset_id, config}
    API->>DB: INSERT experiments (status=queued)
    API->>R: Enqueue run_experiment(experiment_id, config, data_path)
    API-->>ST: 201 {experiment_id, task_id}

    ST->>API: WS /ws/experiments/{id}/progress
    Note over ST,API: WebSocket held open

    R->>W: Deliver run_experiment
    W->>DB: UPDATE experiments (status=running, started_at=now)

    rect rgb(230, 245, 255)
        Note over W: Phase 1 — Understand
        W->>AS: Load dataset
        W->>W: Profile + Validate + Preprocess
        W->>R: PUBLISH progress (phase=understand, 25%)
        R-->>API: Relay to WS
        API-->>ST: WS frame
    end

    rect rgb(230, 255, 230)
        Note over W: Phase 2 — Model
        W->>W: Train Ridge + XGBoost × 5 folds
        W->>AS: Save fitted models
        W->>W: Compute OOF residuals
        W->>AS: Save residual arrays
        W->>R: PUBLISH progress (phase=model, 50%)
    end

    rect rgb(255, 245, 230)
        Note over W: Phase 3 — Detect
        W->>W: HDBSCAN clustering
        W->>W: Subgroup discovery
        W->>W: SHAP interactions
        W->>W: PatternScorer.score_and_rank()
        W->>R: PUBLISH progress (phase=detect, 75%)
    end

    rect rgb(255, 230, 245)
        Note over W: Phase 4 — Construct
        W->>W: VariableSynthesizer
        W->>W: BootstrapValidator (1000×)
        W->>W: CausalChecker
        W->>W: ExplanationGenerator
        W->>DB: INSERT latent_variables (bulk)
        W->>DB: INSERT error_patterns (bulk)
        W->>R: PUBLISH progress (phase=construct, 100%)
    end

    W->>DB: UPDATE experiments (status=completed, completed_at=now)
    R-->>API: WS completion frame
    API-->>ST: WS {status: completed}
    ST-->>U: "Experiment complete! View results."
```

### 5.2 WebSocket Progress Flow

```mermaid
sequenceDiagram
    participant C as Client (Streamlit)
    participant API as FastAPI
    participant R as Redis Pub/Sub
    participant W as Celery Worker

    C->>API: WS CONNECT /ws/experiments/{id}/progress
    API->>R: SUBSCRIBE ive:progress:{id}
    Note over API: Hold WS connection open

    loop For each phase
        W->>R: PUBLISH ive:progress:{id} {phase, pct, msg}
        R-->>API: Message received
        API-->>C: WS TEXT FRAME {"phase":"model","progress_pct":45}
    end

    W->>R: PUBLISH ive:progress:{id} {phase:"complete", pct:100}
    R-->>API: Final message
    API-->>C: WS TEXT FRAME {"status":"completed", "progress_pct":100}
    API->>C: WS CLOSE
    API->>R: UNSUBSCRIBE ive:progress:{id}
```

### 5.3 Celery Job Processing Flow

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant R as Redis Broker
    participant W as Worker Process
    participant DB as PostgreSQL

    API->>R: celery.send_task("run_experiment", args=[exp_id, config, path])
    Note over R: Task in "analysis" queue

    W->>R: Basic.Get (prefetch=1)
    R-->>W: Task message

    alt Task succeeds
        W->>W: Execute run_experiment()
        W->>DB: UPDATE experiment (status=completed)
        W->>R: Basic.Ack (task complete)
        W->>R: Store result (TTL=24h)
    else Task fails (transient)
        W->>W: on_failure() logs error
        W->>R: Basic.Reject + requeue
        Note over R: Retry after 60s (max 2 retries)
    else Task fails (permanent)
        W->>W: on_failure() logs error
        W->>DB: UPDATE experiment (status=failed, error_message)
        W->>R: Basic.Ack (don't requeue)
    else Worker crashes
        Note over R: task_acks_late=True → message still on queue
        R->>W: Re-deliver to another worker
    end
```

### 5.4 Bootstrap Validation Loop

```mermaid
sequenceDiagram
    participant S as VariableSynthesizer
    participant B as BootstrapValidator
    participant C as CausalChecker
    participant E as ExplanationGenerator

    S->>S: Convert patterns → candidates
    loop For each candidate
        S->>B: validate(candidate, X, y)
        loop 1000 bootstrap iterations
            B->>B: Resample with replacement
            B->>B: Re-apply detection rule
            B->>B: Compute effect size
        end
        B-->>S: BootstrapResult {stability, CI, p_value}

        alt stability ≥ 0.70 AND ci_lower > 0
            S->>C: check(candidate, df, target)
            C->>C: Check reverse causality
            C->>C: Check confounding proxy
            C-->>S: CausalResult {plausible, warnings}
            S->>E: generate(candidate, profile)
            E-->>S: explanation_text
            Note over S: candidate.status = "validated"
        else stability < 0.70
            Note over S: candidate.status = "rejected"
        end
    end
    S-->>S: Return validated candidates
```

---

## 6. Error Handling Strategy

### 6.1 Error Taxonomy

All errors in IVE are categorised into three tiers:

| Tier       | Category                     | HTTP Code | Retriable?  | Examples                                                             |
| ---------- | ---------------------------- | --------- | ----------- | -------------------------------------------------------------------- |
| **Tier 1** | Input/Validation Errors      | 400/422   | No          | Missing target column, invalid file format, config out of range      |
| **Tier 2** | Processing Errors            | 500       | Conditional | HDBSCAN convergence failure, SHAP timeout, OOM during model training |
| **Tier 3** | System/Infrastructure Errors | 500/503   | Yes         | Database connection lost, Redis unreachable, S3 write failure        |

### 6.2 Error Response Format

All API errors return a consistent JSON structure:

```json
{
  "error": {
    "code": "DATASET_NOT_FOUND",
    "message": "Dataset a1b2c3d4-... does not exist",
    "details": [],
    "request_id": "req-uuid-123",
    "timestamp": "2026-03-02T10:00:00Z"
  }
}
```

**Error codes** follow the pattern `{ENTITY}_{PROBLEM}`:

| Code                           | HTTP | Meaning                                             |
| ------------------------------ | ---- | --------------------------------------------------- |
| `VALIDATION_ERROR`             | 422  | Request body fails Pydantic validation              |
| `DATASET_NOT_FOUND`            | 404  | Dataset UUID does not exist                         |
| `EXPERIMENT_NOT_FOUND`         | 404  | Experiment UUID does not exist                      |
| `DATASET_NOT_PROFILED`         | 409  | Dataset not yet profiled — cannot create experiment |
| `EXPERIMENT_ALREADY_COMPLETED` | 409  | Cannot cancel a completed experiment                |
| `FILE_TOO_LARGE`               | 413  | Uploaded file exceeds 500 MB                        |
| `UNSUPPORTED_FORMAT`           | 415  | File is not CSV or Parquet                          |
| `RATE_LIMITED`                 | 429  | Rate limit exceeded                                 |
| `PIPELINE_ERROR`               | 500  | ML pipeline failed — check `error_message`          |
| `INTERNAL_ERROR`               | 500  | Unexpected server error                             |

### 6.3 Retry Strategy

**Celery task retries (Tier 3 errors only):**

```python
@celery_app.task(
    bind=True,
    max_retries=2,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,        # max 2 minutes between retries
    retry_jitter=True,
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_experiment(self, experiment_id, config, data_path):
    ...
```

| Retry         | Backoff      | Max wait | Action                                         |
| ------------- | ------------ | -------- | ---------------------------------------------- |
| 1st           | 30s ± jitter | 60s      | Retry automatically                            |
| 2nd           | 60s ± jitter | 120s     | Retry automatically                            |
| Final failure | —            | —        | Mark experiment `status=failed`, log to Sentry |

**Non-retriable errors** (Tier 1 + Tier 2): No retries. The task sets `status=failed` with `error_message` immediately.

### 6.4 Circuit Breaker Pattern

For external dependencies (PostgreSQL, Redis, S3), the system uses a simple in-process circuit breaker:

```
STATE: CLOSED → OPEN → HALF_OPEN → CLOSED

PARAMETERS:
    failure_threshold = 5        # consecutive failures before opening
    recovery_timeout  = 30s      # time in OPEN state before trying HALF_OPEN
    success_threshold = 2        # successes in HALF_OPEN before closing

BEHAVIOUR:
    CLOSED:     Normal operation. Increment failure counter on error.
                If failure_count ≥ failure_threshold → switch to OPEN.
    OPEN:       All calls immediately fail with SERVICE_UNAVAILABLE.
                After recovery_timeout → switch to HALF_OPEN.
    HALF_OPEN:  Allow 1 call through.
                If success → increment success counter.
                If success_count ≥ success_threshold → switch to CLOSED.
                If failure → switch to OPEN immediately.
```

This is implemented in `api/middleware/error_handler.py` and applied to:

- Database session acquisition
- Redis connection pool
- S3 client operations

### 6.5 Phase-Level Error Isolation

Each pipeline phase wraps its execution in a try/except. If a non-critical phase fails (e.g., `TemporalAnalyzer` when no datetime column exists), the engine logs a warning and continues to the next phase. Only Phase 2 (Model) failures are fatal — without residuals, no detection is possible.

```python
for phase in phases:
    try:
        result = await phase.execute(ctx)
    except CriticalPhaseError:
        raise  # Phase 2 failure → abort entire pipeline
    except PhaseError as e:
        logger.warning("phase_skipped", phase=phase.name, error=str(e))
        continue  # Non-critical → skip and continue
```

---

## 7. Configuration Schema

### 7.1 Pydantic Settings Model

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class IVESettings(BaseSettings):
    """Complete application configuration — loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ─────────────────────────────────────────────
    # Application
    # ─────────────────────────────────────────────
    env: str = Field("development", description="Environment name: development | staging | production")
    debug: bool = Field(False, description="Enable debug mode (verbose logging, auto-reload)")
    secret_key: str = Field(..., description="Application-wide secret for signing tokens")
    app_name: str = Field("Invisible Variables Engine", description="Application display name")
    app_version: str = Field("0.1.0", description="Semantic version")

    # ─────────────────────────────────────────────
    # Database (PostgreSQL)
    # ─────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://ive:ive@localhost:5432/ive",
        description="Async database URL"
    )
    db_pool_size: int = Field(5, ge=1, le=20, description="SQLAlchemy connection pool size")
    db_max_overflow: int = Field(10, ge=0, le=50, description="Max connections beyond pool_size")
    db_pool_timeout: int = Field(30, ge=5, le=120, description="Seconds to wait for a connection")
    db_echo: bool = Field(False, description="Echo SQL statements to log")

    # ─────────────────────────────────────────────
    # Redis
    # ─────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    redis_result_ttl: int = Field(86400, ge=3600, description="Celery result TTL in seconds (default 24h)")

    # ─────────────────────────────────────────────
    # Authentication
    # ─────────────────────────────────────────────
    valid_api_keys: str = Field(
        "dev-key-change-me",
        description="Comma-separated list of valid API keys"
    )

    # ─────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────
    rate_limit_per_minute: int = Field(100, ge=1, le=10000, description="Default rate limit per API key")

    # ─────────────────────────────────────────────
    # File Uploads
    # ─────────────────────────────────────────────
    max_upload_size_mb: int = Field(500, ge=1, le=5000, description="Max upload file size in MB")
    allowed_extensions: str = Field(".csv,.parquet", description="Comma-separated allowed file extensions")

    # ─────────────────────────────────────────────
    # Artifact Storage
    # ─────────────────────────────────────────────
    artifact_store_type: str = Field("local", description="Artifact backend: local | s3")
    artifact_base_dir: str = Field("./artifacts", description="Local artifact store base directory")
    s3_bucket: str = Field("", description="S3 bucket name (when artifact_store_type=s3)")
    s3_region: str = Field("us-east-1", description="AWS region for S3")
    s3_endpoint_url: str = Field("", description="Custom S3 endpoint (e.g. MinIO)")

    # ─────────────────────────────────────────────
    # ML Pipeline Defaults
    # ─────────────────────────────────────────────
    default_cv_folds: int = Field(5, ge=2, le=10, description="Default K-fold splits")
    default_model_types: str = Field("linear,xgboost", description="Comma-separated default model types")
    default_random_seed: int = Field(42, description="Global default random seed")

    # Linear model
    linear_alpha: float = Field(1.0, gt=0, description="Ridge regularisation strength")

    # XGBoost
    xgb_n_estimators: int = Field(300, ge=50, le=2000, description="Number of boosting rounds")
    xgb_max_depth: int = Field(6, ge=2, le=15, description="Max tree depth")
    xgb_learning_rate: float = Field(0.1, gt=0, le=1.0, description="Learning rate (eta)")

    # ─────────────────────────────────────────────
    # Detection Thresholds
    # ─────────────────────────────────────────────
    alpha: float = Field(0.05, gt=0, lt=1.0, description="Statistical significance level")
    min_subgroup_coverage: float = Field(0.03, ge=0.01, le=0.5, description="Min fraction of samples for a subgroup")
    min_cluster_size: int = Field(10, ge=5, le=500, description="HDBSCAN min_cluster_size")
    min_samples: int = Field(5, ge=2, le=100, description="HDBSCAN min_samples (core point threshold)")
    residual_weight: float = Field(2.0, ge=0.5, le=10.0, description="Residual up-weighting in HDBSCAN space")
    beam_width: int = Field(10, ge=3, le=50, description="Beam search width for subgroup discovery")
    max_subgroup_depth: int = Field(3, ge=1, le=5, description="Max conjunction depth for subgroup rules")
    shap_sample_size: int = Field(500, ge=50, le=5000, description="SHAP computation row limit")
    min_effect_size: float = Field(0.2, ge=0, description="Min Cohen's d to consider a pattern significant")

    # ─────────────────────────────────────────────
    # Construction & Validation
    # ─────────────────────────────────────────────
    bootstrap_iterations: int = Field(1000, ge=100, le=10000, description="Bootstrap resampling iterations")
    bootstrap_confidence_level: float = Field(0.95, ge=0.80, le=0.99, description="Bootstrap CI level")
    stability_threshold: float = Field(0.70, ge=0.50, le=0.99, description="Min stability for validation")
    max_latent_variables: int = Field(5, ge=1, le=20, description="Max LV candidates to return")
    jaccard_dedup_threshold: float = Field(0.90, ge=0.5, le=1.0, description="Overlap threshold for dedup")

    # ─────────────────────────────────────────────
    # Scoring Weights
    # ─────────────────────────────────────────────
    score_effect_weight: float = Field(0.40, ge=0, le=1.0, description="Weight for effect size in composite score")
    score_stability_weight: float = Field(0.35, ge=0, le=1.0, description="Weight for stability in composite score")
    score_coverage_weight: float = Field(0.25, ge=0, le=1.0, description="Weight for coverage in composite score")

    # ─────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────
    log_level: str = Field("INFO", description="Log level: DEBUG | INFO | WARNING | ERROR")
    log_format: str = Field("json", description="Log format: json (production) | console (dev)")
    sentry_dsn: str = Field("", description="Sentry DSN for error tracking (empty = disabled)")

    # ─────────────────────────────────────────────
    # Celery / Workers
    # ─────────────────────────────────────────────
    celery_broker_url: str = Field("redis://localhost:6379/0", description="Celery broker URL")
    celery_result_backend: str = Field("redis://localhost:6379/1", description="Celery result backend URL")
    celery_worker_concurrency: int = Field(2, ge=1, le=16, description="Worker processes per container")
    celery_task_time_limit: int = Field(3600, ge=60, description="Hard task time limit in seconds (1h default)")
    celery_task_soft_time_limit: int = Field(3300, ge=60, description="Soft limit for graceful shutdown (55min)")

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.valid_api_keys.split(",") if k.strip()]

    @property
    def model_types_list(self) -> list[str]:
        return [m.strip() for m in self.default_model_types.split(",") if m.strip()]
```

### 7.2 Environment File (.env.example)

```bash
# ─── Application ───
ENV=development
DEBUG=true
SECRET_KEY=change-me-in-production
APP_VERSION=0.1.0

# ─── Database ───
DATABASE_URL=postgresql+asyncpg://ive:ive@localhost:5432/ive
DB_POOL_SIZE=5
DB_ECHO=false

# ─── Redis ───
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# ─── Auth ───
VALID_API_KEYS=dev-key-change-me

# ─── Storage ───
ARTIFACT_STORE_TYPE=local
ARTIFACT_BASE_DIR=./artifacts

# ─── Logging ───
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

---

_End of Low-Level Design Document_
