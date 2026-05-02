"""
API v1 Router — Invisible Variables Engine.

Aggregates all v1 endpoint routers into a single router mounted
under ``/api/v1`` in the main application.

Route map::

    GET  /api/v1/health                                → liveness probe
    GET  /api/v1/health/ready                          → readiness probe

    POST /api/v1/datasets                              → upload dataset
    GET  /api/v1/datasets                              → list datasets
    GET  /api/v1/datasets/{id}                         → dataset detail
    DEL  /api/v1/datasets/{id}                         → delete dataset

    POST /api/v1/experiments                           → create experiment
    GET  /api/v1/experiments                           → list experiments
    GET  /api/v1/experiments/{id}                      → experiment detail
    GET  /api/v1/experiments/{id}/progress             → progress poll
    POST /api/v1/experiments/{id}/cancel               → cancel
    DEL  /api/v1/experiments/{id}                      → delete experiment

    GET  /api/v1/experiments/{id}/latent-variables     → list LVs
    GET  /api/v1/experiments/{id}/latent-variables/{v} → LV detail
"""

from __future__ import annotations

from fastapi import APIRouter

from ive.api.v1.endpoints.api_keys import router as api_keys_router
from ive.api.v1.endpoints.column_metadata import router as column_metadata_router
from ive.api.v1.endpoints.datasets import router as datasets_router
from ive.api.v1.endpoints.experiments import router as experiments_router
from ive.api.v1.endpoints.health import router as health_router
from ive.api.v1.endpoints.latent_variables import router as latent_variables_router
from ive.api.v1.endpoints.lv_annotations import router as lv_annotations_router
from ive.api.v1.endpoints.share_tokens import (
    issue_router as share_tokens_issue_router,
    public_router as share_tokens_public_router,
)

api_v1_router = APIRouter()

# Health — no prefix, exempt from auth
api_v1_router.include_router(health_router, tags=["Health"])

# Datasets
api_v1_router.include_router(
    datasets_router,
    prefix="/datasets",
    tags=["Datasets"],
)

# Experiments
api_v1_router.include_router(
    experiments_router,
    prefix="/experiments",
    tags=["Experiments"],
)

# Latent variables — nested under /experiments/{id}/...
api_v1_router.include_router(
    latent_variables_router,
    prefix="/experiments",
    tags=["Latent Variables"],
)

# LV annotations (Phase C2.1) — free-text notes attached to a single LV.
api_v1_router.include_router(
    lv_annotations_router,
    prefix="/latent-variables/{lv_id}/annotations",
    tags=["Latent Variables"],
)

# Share token issuance + management (Phase C2.2) — admin-scoped.
api_v1_router.include_router(
    share_tokens_issue_router,
    prefix="/experiments/{experiment_id}/shares",
    tags=["Sharing"],
)

# Public share-read endpoint (Phase C2.2) — token-gated, bypasses
# X-API-Key auth via path-based exemption in APIKeyMiddleware (legacy
# middleware) and explicit handler-side checks (defense in depth).
api_v1_router.include_router(
    share_tokens_public_router,
    prefix="/share",
    tags=["Sharing"],
)

# Column sensitivity metadata — gated per-dataset
api_v1_router.include_router(
    column_metadata_router,
    prefix="/datasets/{dataset_id}/columns",
    tags=["Datasets"],
)

# API key management — admin scope only
api_v1_router.include_router(
    api_keys_router,
    prefix="/api-keys",
    tags=["Auth"],
)
