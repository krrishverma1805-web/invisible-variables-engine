"""
IVE Database Package.

Re-exports the declarative Base, session management functions, and all
ORM model classes for convenient imports::

    from ive.db import Base, get_session, Dataset, Experiment
"""

from ive.db.database import Base, close_db, get_engine, get_session, init_db
from ive.db.models import (
    APIKey,
    Dataset,
    ErrorPattern,
    Experiment,
    ExperimentEvent,
    LatentVariable,
    Residual,
    TrainedModel,
)

__all__ = [
    # database
    "Base",
    "init_db",
    "close_db",
    "get_session",
    "get_engine",
    # models
    "Dataset",
    "Experiment",
    "TrainedModel",
    "Residual",
    "ErrorPattern",
    "LatentVariable",
    "APIKey",
    "ExperimentEvent",
]
