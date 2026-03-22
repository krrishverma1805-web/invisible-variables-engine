"""
Abstract Base Model for IVE ML Models.

Defines the interface that all IVE-compatible ML models must implement.
This allows the cross-validator and residual analyzer to work with any
model type interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class IVEModel(ABC):
    """
    Abstract base class for all IVE ML models.

    Every model must support: fit, predict, feature importance, and SHAP values.
    """

    @abstractmethod
    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """
        Fit the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Prediction array of shape (n_samples,).
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Return normalised feature importance scores.

        Returns:
            Dict mapping feature name → importance score (0–1).
        """
        ...

    @abstractmethod
    def get_shap_values(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute SHAP values for the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            SHAP value array of shape (n_samples, n_features).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a canonical name for this model type."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return getattr(self, "_fitted", False)

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters as a serialisable dict."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


__all__ = ["IVEModel"]
