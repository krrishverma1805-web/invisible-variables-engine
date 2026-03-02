# User Guide — Invisible Variables Engine

## Getting Started

### What is IVE?

The Invisible Variables Engine helps you discover **hidden features** that your model should know about but doesn't. If your model consistently under-predicts for certain groups of samples, IVE will tell you what those groups have in common — and will name the latent variable that would fix it.

### Typical Workflow

1. **Upload your dataset** (CSV or Parquet)
2. **Configure and start an experiment**
3. **Monitor progress** in real-time
4. **Review discovered latent variables**
5. **Download the explanation report**

---

## Streamlit UI Walkthrough

### Page 1 — Upload Dataset

- Click **"Upload file"** and select your CSV or Parquet file
- Enter a **name** for the dataset
- Specify the **target column** (the column you're trying to predict)
- Click **"Upload & Profile"**
- IVE will ingest the file and run an automatic data profile

> **Tip:** For best results, your dataset should have at least 500 rows and 5 features.

### Page 2 — Configure Experiment

- Select the dataset to analyse
- Choose model types: **Linear** (fast), **XGBoost** (thorough), or both
- Set cross-validation folds (default 5)
- Set the maximum number of latent variables to discover (default 5)
- Click **"Start Analysis"**

The experiment is queued and will run asynchronously.

### Page 3 — Monitor Progress

View real-time progress bars for each phase:

| Phase      | Typical Duration | What's Happening       |
| ---------- | ---------------- | ---------------------- |
| Understand | 10–60s           | Profiling your data    |
| Model      | 30s–10min        | Training models        |
| Detect     | 1–5min           | Finding error clusters |
| Construct  | 30s–2min         | Naming the variables   |

Click **"Refresh"** or enable **"Auto-refresh"** for live updates.

### Page 4 — Results

Each discovered latent variable shows:

- **Name** — AI-generated descriptive name
- **Explanation** — Plain English description of what the variable represents
- **Confidence Score** — How strongly we believe this is real (0–1)
- **Effect Size** — How much this variable explains the model error (Cohen's d)
- **Coverage** — What fraction of samples are affected
- **Candidate Features** — Existing features that are proxies for this variable
- **Validation** — Bootstrap stability and p-value

#### Interpreting Results

| Confidence | Interpretation                                 |
| ---------- | ---------------------------------------------- |
| > 0.8      | Strong evidence; worth collecting this feature |
| 0.6–0.8    | Moderate evidence; investigate further         |
| < 0.6      | Weak evidence; may be noise                    |

---

## API Usage

### Authentication

```bash
export API_KEY="your-api-key"
export BASE_URL="http://localhost:8000/api/v1"
```

### Upload a Dataset

```bash
curl -X POST "$BASE_URL/datasets" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/housing.csv" \
  -F "name=Housing Prices" \
  -F "target_column=price"
```

### Start an Experiment

```bash
curl -X POST "$BASE_URL/experiments" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset-uuid>",
    "name": "Housing Analysis v1",
    "config": {
      "model_types": ["linear", "xgboost"],
      "cv_folds": 5
    }
  }'
```

### Poll for Results

```bash
# Check experiment status
curl "$BASE_URL/experiments/<experiment-uuid>" \
  -H "X-API-Key: $API_KEY"

# Get discovered latent variables
curl "$BASE_URL/experiments/<experiment-uuid>/latent-variables" \
  -H "X-API-Key: $API_KEY"
```

---

## FAQ

**Q: How large can my dataset be?**
A: IVE supports up to 1 million rows and 500 features. Larger datasets will take longer to process.

**Q: What file formats are supported?**
A: CSV and Parquet. For large datasets, Parquet is recommended for faster ingestion.

**Q: What types of tasks are supported?**
A: Currently regression and binary classification. Multi-class classification is planned.

**Q: How do I interpret a p-value < 0.05 for a latent variable?**
A: It means the discovered subgroup pattern is statistically unlikely to have occurred by chance at the 5% significance level. Treat it as a starting point for further investigation, not definitive proof.

**Q: Can I run IVE programmatically in my own pipeline?**
A: Yes — see the Python SDK section below. You can also import `ive.core.engine.IVEEngine` directly.

---

## Python SDK Quick Start

```python
import asyncio
from ive.core.engine import IVEEngine
from ive.core.pipeline import ExperimentConfig

async def main():
    engine = IVEEngine()
    config = ExperimentConfig(
        model_types=["xgboost"],
        cv_folds=5,
        random_seed=42,
    )
    result = await engine.run(experiment_id=..., config=config)
    for lv in result.latent_variables:
        print(f"{lv.name}: {lv.explanation}")

asyncio.run(main())
```
