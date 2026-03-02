"""
API v1 Router.

Aggregates all v1 endpoint routers into a single router that is mounted
under /api/v1 in the main application.
"""

from __future__ import annotations

from fastapi import APIRouter

from ive.api.v1.endpoints.datasets import router as datasets_router
from ive.api.v1.endpoints.experiments import router as experiments_router
from ive.api.v1.endpoints.health import router as health_router
from ive.api.v1.endpoints.latent_variables import router as latent_variables_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_v1_router.include_router(experiments_router, prefix="/experiments", tags=["experiments"])
api_v1_router.include_router(
    latent_variables_router, prefix="/latent-variables", tags=["latent-variables"]
)
