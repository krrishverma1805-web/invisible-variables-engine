"""
FastAPI Service for Invisible Variables Engine.

Exposes endpoints to:
1. Trigger the pipeline.
2. Retrieve discovered latent variables.
3. Fetch explanations.
4. View the full HTML report.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import json
import os
import subprocess
from typing import List, Dict, Any

app = FastAPI(
    title="Invisible Variables Engine API",
    description="API for discovering and explaining invisible variables in machine learning models.",
    version="1.0.0"
)

# Paths
DATA_DIR = "data/outputs"
LATENT_SCORES_PATH = os.path.join(DATA_DIR, "latent_variables/latent_scores.csv")
EXPLANATIONS_PATH = os.path.join(DATA_DIR, "latent_variables/explanations.json")
REPORT_PATH = os.path.join(DATA_DIR, "reports/IVE_Report.html")

# Scripts in execution order
PIPELINE_SCRIPTS = [
    "src/data_validation/leakage_checks.py",
    "src/modeling/train_model.py",
    "src/modeling/confidence.py",
    "src/residuals/compute_residuals.py",
    "src/residuals/filter_residuals.py",
    "src/residuals/stability.py",
    "src/pattern_discovery/clustering.py",
    "src/pattern_discovery/pattern_validation.py",
    "src/latent_engine/latent_construction.py",
    "src/validation/proxy_validation.py",
    "src/validation/retraining_test.py",
    "src/explainability/explanation_generator.py",
    "src/explainability/visualizations.py",
    "src/reporting/report_builder.py"
]

@app.get("/health")
def health_check():
    """Returns the health status of the API."""
    return {"status": "active", "service": "Invisible Variables Engine"}

def run_pipeline_task():
    """Runs the full pipeline sequence."""
    print("Starting pipeline execution...")
    try:
        for script in PIPELINE_SCRIPTS:
            print(f"Running {script}...")
            subprocess.run(["python3", script], check=True)
        print("Pipeline execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed at {script}: {e}")
    except Exception as e:
        print(f"Pipeline error: {e}")

@app.post("/run")
def run_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers the full Invisible Variables Engine pipeline in the background.
    """
    background_tasks.add_task(run_pipeline_task)
    return {"message": "Pipeline triggered successfully. Check logs for progress."}

@app.get("/latent-variables")
def get_latent_variables():
    """Returns the constructed latent variable scores."""
    if not os.path.exists(LATENT_SCORES_PATH):
        raise HTTPException(status_code=404, detail="Latent variables not found. Run pipeline first.")
    
    try:
        df = pd.read_csv(LATENT_SCORES_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explanations")
def get_explanations():
    """Returns the generated human-readable explanations."""
    if not os.path.exists(EXPLANATIONS_PATH):
        raise HTTPException(status_code=404, detail="Explanations not found. Run pipeline first.")
    
    try:
        with open(EXPLANATIONS_PATH, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report", response_class=HTMLResponse)
def get_report():
    """Returns the full HTML forensic report."""
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not found. Run pipeline first.")
    
    try:
        with open(REPORT_PATH, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
