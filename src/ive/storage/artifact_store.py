"""
Artifact Store — Invisible Variables Engine.

Async abstraction over local filesystem (aiofiles) and AWS S3 (aiobotocore)
for storing and retrieving binary artefacts: uploaded datasets, trained model
pickles, NumPy residual arrays, JSON profiles, and HTML/PDF reports.

NOTE — dependency:
    ``aiofiles`` must be in ``pyproject.toml`` (under [tool.poetry.dependencies]):

        aiofiles = ">=23.0,<25.0"

    Add it with:  poetry add aiofiles

Storage layout
--------------
Local (ARTIFACT_STORE_TYPE=local):

    {base_dir}/{category}/{experiment_id}/{unique_filename}
    {base_dir}/{category}/{unique_filename}          # when no experiment_id

S3 (ARTIFACT_STORE_TYPE=s3):

    {category}/{experiment_id}/{unique_filename}
    {category}/{unique_filename}

Categories: ``datasets`` | ``models`` | ``results`` | ``exports``

Unique filename convention::

    {sanitised_stem}_{uuid4_short}.{extension}
    # e.g.  housing_data_a3f9b2c1.csv

Usage::

    from ive.storage.artifact_store import get_artifact_store

    store = get_artifact_store()
    path = await store.save_file(content, category="datasets", filename="data.csv",
                                 experiment_id="exp-uuid")
    data = await store.load_file(path)
    await store.delete_file(path)
"""

from __future__ import annotations

import hashlib
import io
import pickle
import re
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, AsyncIterator, cast

import numpy as np

from ive.config import get_settings
from ive.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Maximum upload size (500 MB default)
# ---------------------------------------------------------------------------
_MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_checksum(file_content: bytes) -> str:
    """Compute a SHA-256 hex digest of ``file_content``.

    Args:
        file_content: Raw bytes to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(file_content).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Remove characters that are unsafe in filesystem paths and S3 keys.

    Keeps: alphanumerics, hyphens, underscores, and dots.
    Replaces everything else with an underscore, then collapses consecutive
    underscores and strips leading/trailing underscores.

    Args:
        filename: Original filename (may include directory separators).

    Returns:
        Safe, normalised filename string.
    """
    # Take basename only — drop directory components
    name = Path(filename).name
    # Replace non-safe characters
    name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    # Collapse runs of underscores/dots
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "artifact"


def get_file_extension(filename: str) -> str:
    """Return the lowercase file extension (without the leading dot).

    Args:
        filename: Filename or path.

    Returns:
        Extension string, e.g. ``"csv"``.  Empty string if no extension.
    """
    suffix = Path(filename).suffix
    return suffix.lstrip(".").lower()


def _make_unique_filename(filename: str) -> str:
    """Generate a collision-safe filename by appending a UUID4 short token.

    Example::

        "housing data.csv"  →  "housing_data_a3f9b2c1.csv"

    Args:
        filename: Original filename.

    Returns:
        Unique, sanitised filename.
    """
    safe = sanitize_filename(filename)
    stem = Path(safe).stem
    ext = Path(safe).suffix  # includes the dot
    short_id = uuid.uuid4().hex[:8]
    return f"{stem}_{short_id}{ext}"


def _resolve_path_parts(
    category: str,
    filename: str,
    experiment_id: str | None,
) -> str:
    """Build the canonical logical path for an artefact.

    Args:
        category:      Artefact category (``"datasets"``, ``"models"``, …).
        filename:      Unique filename (already made collision-safe).
        experiment_id: Optional parent experiment UUID.

    Returns:
        Forward-slash-delimited logical path.
    """
    if experiment_id:
        return f"{category}/{experiment_id}/{filename}"
    return f"{category}/{filename}"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ArtifactStore(ABC):
    """Abstract async interface for persistent artefact storage.

    All methods are ``async`` so that both the local (aiofiles) and S3
    (aiobotocore) backends can perform non-blocking I/O without blocking the
    FastAPI event loop.
    """

    # -- Core abstract methods -----------------------------------------------

    @abstractmethod
    async def save_file(
        self,
        file_content: bytes | BinaryIO,
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        """Persist a file and return its canonical storage path/key.

        Args:
            file_content:  Raw bytes or a binary file-like object.
            category:      Artefact category: ``"datasets"``, ``"models"``,
                           ``"results"``, or ``"exports"``.
            filename:      Original filename (will be sanitised + made unique).
            experiment_id: Optional parent experiment UUID; used to namespace
                           the file under the correct experiment directory.

        Returns:
            The storage path (filesystem path for local; S3 key for S3).

        Raises:
            ValueError: If ``file_content`` exceeds the configured max size.
        """
        ...

    @abstractmethod
    async def load_file(self, path: str) -> bytes:
        """Load and return the full contents of a stored artefact.

        Args:
            path: The path/key returned by :meth:`save_file`.

        Returns:
            Raw bytes.

        Raises:
            FileNotFoundError: If no artefact exists at ``path``.
        """
        ...

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete a stored artefact.

        Args:
            path: The path/key returned by :meth:`save_file`.

        Returns:
            ``True`` if the artefact existed and was deleted.
        """
        ...

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Return ``True`` if an artefact exists at ``path``.

        Args:
            path: The path/key returned by :meth:`save_file`.
        """
        ...

    @abstractmethod
    async def get_file_size(self, path: str) -> int:
        """Return the stored size of an artefact in bytes.

        Args:
            path: The path/key returned by :meth:`save_file`.

        Raises:
            FileNotFoundError: If no artefact exists at ``path``.
        """
        ...

    @abstractmethod
    async def list_files(self, prefix: str) -> list[str]:
        """List all artefact paths whose key begins with ``prefix``.

        Args:
            prefix: Path prefix, e.g. ``"experiments/abc/"`` or ``"datasets/"``.

        Returns:
            Sorted list of matching paths/keys.
        """
        ...

    # -- Composite helpers (shared across both backends) ---------------------

    async def save_json(
        self,
        data: object,
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        """Serialise ``data`` as UTF-8 JSON and persist.

        Args:
            data:          JSON-serialisable Python object.
            category:      Artefact category.
            filename:      Target filename (will be sanitised).
            experiment_id: Optional experiment UUID.

        Returns:
            Storage path.
        """
        import json

        def _default(obj: object) -> object:
            import uuid as _uuid
            from datetime import date, datetime

            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, _uuid.UUID):
                return str(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

        encoded = json.dumps(data, default=_default, ensure_ascii=False).encode("utf-8")
        return await self.save_file(encoded, category, filename, experiment_id)

    async def load_json(self, path: str) -> object:
        """Load and deserialise a JSON artefact.

        Args:
            path: Storage path/key.

        Returns:
            Deserialised Python object.
        """
        import json

        raw = await self.load_file(path)
        return json.loads(raw.decode("utf-8"))

    async def save_numpy(
        self,
        array: np.ndarray[Any, Any],
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        """Serialise a NumPy array with ``numpy.save`` and persist.

        Args:
            array:         Array to store.
            category:      Artefact category.
            filename:      Target filename (should end in ``.npy``).
            experiment_id: Optional experiment UUID.

        Returns:
            Storage path.
        """
        buf = io.BytesIO()
        np.save(buf, array, allow_pickle=False)
        return await self.save_file(buf.getvalue(), category, filename, experiment_id)

    async def load_numpy(self, path: str) -> np.ndarray[Any, Any]:
        """Load a NumPy ``.npy`` artefact.

        Args:
            path: Storage path/key.

        Returns:
            NumPy array.
        """
        raw = await self.load_file(path)
        return cast(np.ndarray[Any, Any], np.load(io.BytesIO(raw), allow_pickle=False))

    async def save_pickle(
        self,
        obj: object,
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        """Pickle ``obj`` with protocol 4 and persist.

        Args:
            obj:           Python object to pickle (e.g. an ML model).
            category:      Artefact category.
            filename:      Target filename.
            experiment_id: Optional experiment UUID.

        Returns:
            Storage path.
        """
        return await self.save_file(
            pickle.dumps(obj, protocol=4), category, filename, experiment_id
        )

    async def load_pickle(self, path: str) -> object:
        """Load and unpickle an artefact.

        Args:
            path: Storage path/key.

        Returns:
            Unpickled Python object.
        """
        return pickle.loads(await self.load_file(path))  # noqa: S301

    async def delete_experiment_files(self, experiment_id: str) -> int:
        """Delete all artefacts belonging to a given experiment.

        Iterates over every category and deletes matching files.

        Args:
            experiment_id: UUID of the experiment to purge.

        Returns:
            Total number of files deleted.
        """
        categories = ("datasets", "models", "results", "exports")
        deleted = 0
        for cat in categories:
            prefix = f"{cat}/{experiment_id}/"
            for path in await self.list_files(prefix):
                if await self.delete_file(path):
                    deleted += 1
        logger.info(
            "artifact.experiment_purged",
            experiment_id=experiment_id,
            files_deleted=deleted,
        )
        return deleted


# ---------------------------------------------------------------------------
# Local filesystem implementation
# ---------------------------------------------------------------------------


class LocalArtifactStore(ArtifactStore):
    """Async filesystem-backed artefact store using ``aiofiles``.

    Files are stored under::

        {base_dir}/{category}/{experiment_id}/{unique_filename}
        {base_dir}/{category}/{unique_filename}        # no experiment_id

    Args:
        base_dir:      Root directory.  Created if absent.
        max_file_size: Maximum bytes allowed per file (default 500 MB).
    """

    def __init__(
        self,
        base_dir: str,
        max_file_size: int = _MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        logger.info(
            "artifact_store.local.ready",
            base_dir=str(self.base_dir),
            max_file_size_mb=max_file_size // (1024 * 1024),
        )

    # -- Internal helpers ----------------------------------------------------

    def _abs(self, logical_path: str) -> Path:
        """Resolve a logical path to an absolute filesystem path.

        Raises:
            ValueError: On path traversal attempts.
        """
        safe = logical_path.lstrip("/").replace("..", "")
        resolved = (self.base_dir / safe).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: logical_path={logical_path!r}")
        return resolved

    def _validate_size(self, data: bytes) -> None:
        if len(data) > self.max_file_size:
            raise ValueError(
                f"File size {len(data):,} bytes exceeds maximum "
                f"{self.max_file_size:,} bytes ({self.max_file_size // (1024 * 1024)} MB)"
            )

    # -- Abstract implementations -------------------------------------------

    async def save_file(
        self,
        file_content: bytes | BinaryIO,
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        try:
            import aiofiles  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "aiofiles is required.  Add it to pyproject.toml: " "poetry add aiofiles"
            ) from exc

        # Materialise BinaryIO to bytes
        if not isinstance(file_content, bytes):
            file_content = file_content.read()

        self._validate_size(file_content)

        unique_name = _make_unique_filename(filename)
        logical = _resolve_path_parts(category, unique_name, experiment_id)
        abs_path = self._abs(logical)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(abs_path, "wb") as fh:
            await fh.write(file_content)

        checksum = compute_checksum(file_content)
        logger.debug(
            "artifact.saved",
            path=logical,
            size_bytes=len(file_content),
            checksum=checksum[:12] + "...",
            category=category,
            experiment_id=experiment_id,
        )
        # Return the absolute path so callers can store it in the DB
        return str(abs_path)

    async def load_file(self, path: str) -> bytes:
        try:
            import aiofiles
        except ImportError as exc:
            raise ImportError("aiofiles is required.  poetry add aiofiles") from exc

        abs_path = Path(path) if Path(path).is_absolute() else self._abs(path)
        if not abs_path.exists():
            logger.error("artifact.not_found", path=str(abs_path))
            raise FileNotFoundError(f"Artefact not found: {abs_path}")

        async with aiofiles.open(abs_path, "rb") as fh:
            data = await fh.read()

        logger.debug("artifact.loaded", path=str(abs_path), size_bytes=len(data))
        return cast(bytes, data)

    async def delete_file(self, path: str) -> bool:
        abs_path = Path(path) if Path(path).is_absolute() else self._abs(path)
        if abs_path.is_file():
            abs_path.unlink()
            logger.debug("artifact.deleted", path=str(abs_path))
            return True
        return False

    async def file_exists(self, path: str) -> bool:
        abs_path = Path(path) if Path(path).is_absolute() else self._abs(path)
        return abs_path.is_file()

    async def get_file_size(self, path: str) -> int:
        abs_path = Path(path) if Path(path).is_absolute() else self._abs(path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Artefact not found: {abs_path}")
        return abs_path.stat().st_size

    async def list_files(self, prefix: str = "") -> list[str]:
        search_root = self._abs(prefix) if prefix else self.base_dir
        if not search_root.exists():
            return []
        return sorted(str(p) for p in search_root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# S3 implementation (stub — ready for aiobotocore in production)
# ---------------------------------------------------------------------------


class S3ArtifactStore(ArtifactStore):
    """AWS S3-backed artefact store using ``aiobotocore``.

    ``aiobotocore`` is the async-native wrapper around ``botocore`` and is
    compatible with ``asyncio`` / FastAPI event loops.

    Add to pyproject.toml before using::

        aiobotocore = {extras = ["boto3"], version = ">=2.0"}

    Args:
        bucket:        S3 bucket name.
        region:        AWS region (default ``"us-east-1"``).
        prefix:        Optional key namespace within the bucket.
        endpoint_url:  Custom S3-compatible endpoint (e.g. MinIO).
        max_file_size: Maximum bytes allowed per upload.
    """

    def __init__(
        self,
        bucket: str,
        *,
        region: str = "us-east-1",
        prefix: str = "",
        endpoint_url: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        max_file_size: int = _MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.endpoint_url = endpoint_url
        self._access_key = aws_access_key
        self._secret_key = aws_secret_key
        self.max_file_size = max_file_size

    def _full_key(self, logical: str) -> str:
        return self.prefix + logical.lstrip("/")

    def _extract_key(self, path: str) -> str:
        """Extract the S3 object key from an ``s3://`` URI or logical path.

        If ``path`` starts with ``s3://{bucket}/``, the bucket prefix is
        stripped and the raw key is returned.  Otherwise, ``path`` is treated
        as a logical path and run through :meth:`_full_key`.
        """
        s3_prefix = f"s3://{self.bucket}/"
        if path.startswith(s3_prefix):
            return path[len(s3_prefix):]
        return self._full_key(path)

    @asynccontextmanager
    async def _s3_client(self) -> AsyncIterator[Any]:
        """Create an aiobotocore S3 client as an async context manager."""
        import aiobotocore.session  # lazy import

        session = aiobotocore.session.AioSession()
        kwargs: dict[str, Any] = {
            "region_name": self.region,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self._access_key:
            kwargs["aws_access_key_id"] = self._access_key
        if self._secret_key:
            kwargs["aws_secret_access_key"] = self._secret_key
        async with session.create_client("s3", **kwargs) as client:
            yield client

    async def save_file(
        self,
        file_content: bytes | BinaryIO,
        category: str,
        filename: str,
        experiment_id: str | None = None,
    ) -> str:
        if not isinstance(file_content, bytes):
            file_content = file_content.read()

        if len(file_content) > self.max_file_size:
            raise ValueError(
                f"File size {len(file_content):,} bytes exceeds maximum "
                f"{self.max_file_size:,} bytes ({self.max_file_size // (1024 * 1024)} MB)"
            )

        unique_name = _make_unique_filename(filename)
        logical = _resolve_path_parts(category, unique_name, experiment_id)
        key = self._full_key(logical)

        async with self._s3_client() as client:
            await client.put_object(Bucket=self.bucket, Key=key, Body=file_content)

        logger.info("s3.save", key=key, size=len(file_content))
        return f"s3://{self.bucket}/{key}"

    async def load_file(self, path: str) -> bytes:
        key = self._extract_key(path)

        async with self._s3_client() as client:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            data: bytes = await response["Body"].read()

        logger.info("s3.load", key=key, size=len(data))
        return data

    async def delete_file(self, path: str) -> bool:
        key = self._extract_key(path)

        async with self._s3_client() as client:
            await client.delete_object(Bucket=self.bucket, Key=key)

        logger.info("s3.delete", key=key)
        return True

    async def file_exists(self, path: str) -> bool:
        key = self._extract_key(path)

        async with self._s3_client() as client:
            try:
                await client.head_object(Bucket=self.bucket, Key=key)
                return True
            except client.exceptions.NoSuchKey:
                return False
            except Exception as exc:
                # Only treat 404 ClientError as "not found"; re-raise everything else
                try:
                    from botocore.exceptions import ClientError
                    if isinstance(exc, ClientError) and exc.response["Error"]["Code"] == "404":
                        return False
                except (ImportError, KeyError, TypeError):
                    pass
                logger.error("s3.file_exists_error", key=key, error=str(exc))
                raise

    async def get_file_size(self, path: str) -> int:
        key = self._extract_key(path)

        async with self._s3_client() as client:
            response = await client.head_object(Bucket=self.bucket, Key=key)
            return int(response["ContentLength"])

    async def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        keys: list[str] = []

        async with self._s3_client() as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.bucket, Prefix=full_prefix
            ):
                for obj in page.get("Contents", []):
                    logical = obj["Key"]
                    if self.prefix and logical.startswith(self.prefix):
                        logical = logical[len(self.prefix):]
                    keys.append(logical)

        return keys


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_artifact_store() -> ArtifactStore:
    """Return the configured ``ArtifactStore`` for this process.

    Reads ``ARTIFACT_STORE_TYPE``, ``ARTIFACT_BASE_DIR``, ``S3_BUCKET_NAME``,
    and related AWS settings from :class:`ive.config.Settings`.

    Returns:
        :class:`LocalArtifactStore` when ``ARTIFACT_STORE_TYPE=local``.
        :class:`S3ArtifactStore`    when ``ARTIFACT_STORE_TYPE=s3``.

    Raises:
        ValueError:  If store type is ``"s3"`` but ``S3_BUCKET_NAME`` is empty.

    Note:
        The factory is **not** cached because the store is lightweight to
        construct and tests need to override config without cache invalidation.
        If you need a singleton, wrap in ``@lru_cache(maxsize=1)`` at the
        call site.
    """
    settings = get_settings()

    if settings.artifact_store_type == "s3":
        if not settings.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME must be configured when ARTIFACT_STORE_TYPE=s3")
        secret = settings.aws_secret_access_key.get_secret_value()
        return S3ArtifactStore(
            bucket=settings.s3_bucket_name,
            region=settings.aws_region,
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key=settings.aws_access_key_id or None,
            aws_secret_key=secret or None,
            max_file_size=settings.max_upload_size_mb * 1024 * 1024,
        )

    return LocalArtifactStore(
        base_dir=settings.artifact_base_dir,
        max_file_size=settings.max_upload_size_mb * 1024 * 1024,
    )
