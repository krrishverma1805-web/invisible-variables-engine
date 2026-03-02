"""
Artifact Store.

Abstracts over local filesystem and S3 for storing binary artifacts
(pickled models, NumPy arrays, JSON profiles, PDF reports).

Factory function `get_artifact_store()` returns the correct implementation
based on `settings.artifact_store_type`.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ive.config import get_settings

log = structlog.get_logger(__name__)


class ArtifactStore(ABC):
    """Abstract interface for persistent artifact storage."""

    @abstractmethod
    def save_bytes(self, key: str, data: bytes) -> str:
        """Persist raw bytes under a logical key. Returns the storage path."""
        ...

    @abstractmethod
    def load_bytes(self, key: str) -> bytes:
        """Load bytes from storage by key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if an artifact with the given key exists."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an artifact. Returns True if deleted."""
        ...

    def save_json(self, key: str, data: Any) -> str:
        """Serialise data as JSON and persist."""
        return self.save_bytes(key, json.dumps(data, default=str).encode("utf-8"))

    def load_json(self, key: str) -> Any:
        """Load and deserialise JSON artifact."""
        return json.loads(self.load_bytes(key).decode("utf-8"))

    def save_numpy(self, key: str, array: np.ndarray) -> str:
        """Serialise and persist a NumPy array."""
        import io
        buf = io.BytesIO()
        np.save(buf, array, allow_pickle=False)
        return self.save_bytes(key, buf.getvalue())

    def load_numpy(self, key: str) -> np.ndarray:
        """Load a NumPy array from storage."""
        import io
        raw = self.load_bytes(key)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def save_pickle(self, key: str, obj: Any) -> str:
        """Pickle and persist an object (use sparingly for ML models)."""
        return self.save_bytes(key, pickle.dumps(obj, protocol=4))

    def load_pickle(self, key: str) -> Any:
        """Load and unpickle an object."""
        return pickle.loads(self.load_bytes(key))


class LocalArtifactStore(ArtifactStore):
    """Stores artifacts on the local filesystem."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Resolve key to absolute filesystem path."""
        p = (self.base_dir / key).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.write_bytes(data)
        log.debug("ive.artifact.saved", key=key, bytes=len(data))
        return str(path)

    def load_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False


class S3ArtifactStore(ArtifactStore):
    """
    Stores artifacts in an AWS S3 bucket.

    TODO:
        - Import boto3 and initialise an s3_client in __init__
        - Implement save_bytes: s3_client.put_object(Bucket=..., Key=key, Body=data)
        - Implement load_bytes: s3_client.get_object(Bucket=..., Key=key)['Body'].read()
        - Implement exists: head_object(...)
        - Implement delete: delete_object(...)
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region
        # TODO: self.client = boto3.client("s3", region_name=region)

    def save_bytes(self, key: str, data: bytes) -> str:
        raise NotImplementedError("S3ArtifactStore.save_bytes not yet implemented")

    def load_bytes(self, key: str) -> bytes:
        raise NotImplementedError("S3ArtifactStore.load_bytes not yet implemented")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("S3ArtifactStore.exists not yet implemented")

    def delete(self, key: str) -> bool:
        raise NotImplementedError("S3ArtifactStore.delete not yet implemented")


def get_artifact_store() -> ArtifactStore:
    """
    Return the configured ArtifactStore instance.

    Uses settings.artifact_store_type to determine the backend:
        'local' → LocalArtifactStore
        's3'    → S3ArtifactStore
    """
    settings = get_settings()
    if settings.artifact_store_type == "s3":
        if not settings.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME must be set when ARTIFACT_STORE_TYPE=s3")
        return S3ArtifactStore(
            bucket=settings.s3_bucket_name,
            region=settings.aws_region,
        )
    return LocalArtifactStore(base_dir=settings.artifact_base_dir)
