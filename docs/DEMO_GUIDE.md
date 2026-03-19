# Demo Guide

This guide walks through running an IVE demo end-to-end — from dataset upload to latent variable discovery — using the Streamlit UI or the API directly.

---

## Recommended Demo Dataset

Start with **`delivery_hidden_weather.csv`** — it produces the clearest results:

| Property | Value |
|----------|-------|
| **Dataset** | `demo_datasets/delivery_hidden_weather.csv` |
| **Target column** | `delivery_time` |
| **Hidden variable** | Storm delay zone |
| **Rule** | `distance_miles > 10` AND `traffic_index > 7.5` → +25 min delivery time |
| **Affected rows** | ~19.5% |
| **Detection type** | Subgroup |
| **Why it works well** | Large effect size (+25 min), clear subgroup boundary, 19.5% support is comfortably within the validation window |

---

## Recommended Mode

Use **demo mode** (`analysis_mode: "demo"`) for all synthetic datasets. Demo mode uses relaxed bootstrap thresholds specifically calibrated for these datasets:

- Stability threshold: 50% (vs. 70% in production)
- Variance floor: 1e-7 (vs. 1e-5)
- Support window: 0.5%–98% (vs. 1%–95%)

This ensures the hidden variables in the synthetic data are reliably validated during demonstrations.

---

## Demo via Streamlit

### 1. Start services

```bash
make dev
```

### 2. Open Streamlit

Navigate to [http://localhost:8501](http://localhost:8501).

### 3. Upload dataset

- Click the **Upload** section in the sidebar.
- Select `demo_datasets/delivery_hidden_weather.csv`.
- Set **Target column** to `delivery_time`.
- Click **Upload**.

### 4. Launch experiment

- Navigate to the **Experiments** section.
- Select the uploaded dataset.
- Set **Analysis Mode** to `Demo`.
- Click **Run Experiment**.

### 5. Monitor progress

The progress bar updates in real time:
- **5%** — Loading and profiling data
- **20%** — Training models (Linear + XGBoost with 5-fold CV)
- **55%** — Detecting patterns in residuals
- **75%** — Synthesizing latent variable candidates
- **100%** — Bootstrap validation complete

### 6. View results

Once complete, the results page displays:
- **Headline** — e.g., "3 hidden variables discovered in delivery_hidden_weather"
- **Validated Variables** — each with a stability score, explanation, and recommendation
- **Rejected Variables** — with rejection reasons (low variance, support too sparse, etc.)
- **Error Patterns** — raw statistical patterns from Phase 3

---

## Demo via API (curl)

### Upload

```bash
curl -X POST http://localhost:8000/api/v1/datasets/ \
  -H "X-API-Key: dev-key-1" \
  -F "file=@demo_datasets/delivery_hidden_weather.csv" \
  -F "target_column=delivery_time" \
  -F "name=Delivery Weather Demo"
```

Save the `id` from the response.

### Create experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiments/ \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "<DATASET_ID>", "config": {"analysis_mode": "demo"}}'
```

### Poll progress

```bash
watch -n 2 'curl -s -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/<EXPERIMENT_ID>/progress | python3 -m json.tool'
```

### View summary

```bash
curl -H "X-API-Key: dev-key-1" \
  http://localhost:8000/api/v1/experiments/<EXPERIMENT_ID>/summary | python3 -m json.tool
```

### View validated latent variables

```bash
curl -H "X-API-Key: dev-key-1" \
  "http://localhost:8000/api/v1/experiments/<EXPERIMENT_ID>/latent-variables?status=validated" \
  | python3 -m json.tool
```

---

## Showing Flower

While the experiment runs, open [http://localhost:5555](http://localhost:5555) (login: `admin` / `flowerpass`) to show:

- The active `run_experiment` task in the **Tasks** tab
- Worker resource utilization in the **Dashboard** tab
- Task execution timeline and duration after completion

This is useful for demonstrating the asynchronous architecture and Celery integration.

---

## What to Expect

For the `delivery_hidden_weather` dataset in demo mode, expect:

1. **3–5 error patterns** detected — primarily on `distance_miles` and `traffic_index` in their high-value quantile bins.
2. **2–5 latent variable candidates** synthesized.
3. **At least 1–3 candidates validated** with bootstrap presence rates ≥ 50%.
4. **Validated variables** will be named like `Latent_Subgroup_distance_miles_X_Y` and/or `Latent_Subgroup_traffic_index_X_Y`, corresponding to the hidden storm delay zone.
5. **Explanations** will describe an unrecorded condition associated with high distance and traffic, linked to systematic prediction errors.

---

## The Latent Variable Discovery Narrative

When presenting results, frame the story as follows:

1. **The Setup** — "We have a delivery dataset with features like distance, traffic index, and time of day. The model predicts delivery time reasonably well, but not perfectly."

2. **The Blind Spot** — "When we analyze the model's prediction errors, we find they're not random. Certain combinations of distance and traffic have systematically higher errors — the model consistently under-predicts delivery time for those routes."

3. **The Discovery** — "IVE identified that when distance exceeds 10 miles and traffic index exceeds 7.5, deliveries take about 25 minutes longer than expected. This corresponds to a hidden factor — a storm delay zone — that the data doesn't explicitly record."

4. **The Validation** — "This isn't just a fluke. The pattern was confirmed across 50 bootstrap resamples with a 100% presence rate, meaning the construction rule consistently produced the same signal across randomized variants of the data."

5. **The Action** — "Now that we know this hidden variable exists, we can investigate: is there weather data we should be collecting? Are there storm zones along certain routes? Adding this information to the model could significantly improve delivery time predictions."

---

## Other Demo Datasets

| Dataset | Target | What IVE Should Find |
|---------|--------|--------------------|
| `healthcare_hidden_risk` | `recovery_days` | Post-surgery complication linked to high BMI + high blood pressure |
| `manufacturing_hidden_shift` | `defect_rate` | Night shift instability linked to high vibration + high humidity |
| `retail_hidden_promo` | `spend_amount` | Premium promo eligibility linked to high loyalty + large basket size |
| `no_hidden_random_noise` | `target` | No significant patterns — control dataset (validates no false positives) |
