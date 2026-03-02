"""
Data Ingestion Module.

Responsible for loading datasets from various file formats (CSV, Parquet)
into a standardised Polars/Pandas DataFrame that the pipeline can use.

Supported formats:
    - CSV (with automatic delimiter detection)
    - Parquet (Apache format)

Design: Reading is done lazily where possible (Polars lazy frames) to
avoid loading the entire dataset into memory before we know we need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import structlog

log = structlog.get_logger(__name__)

try:
    import polars as pl
    import pandas as pd
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class DataIngestion:
    """
    Loads datasets from disk into a Pandas DataFrame.

    Supports CSV and Parquet formats. Large files are read in chunks
    where possible to keep memory usage bounded.
    """

    _SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}

    def __init__(self, max_rows: int | None = None) -> None:
        """
        Initialise the ingestion handler.

        Args:
            max_rows: Optional row limit (useful for testing and profiling).
        """
        self.max_rows = max_rows

    def load(self, path: Union[str, Path]) -> "pd.DataFrame":
        """
        Load a dataset from disk.

        Args:
            path: Absolute path to the CSV or Parquet file.

        Returns:
            A Pandas DataFrame with the dataset contents.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the path does not exist.

        TODO:
            - Detect delimiter automatically for CSV
            - Use Polars for large files and convert to Pandas at the end
            - Apply max_rows limit if set
            - Log row/column count after loading
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {self._SUPPORTED_EXTENSIONS}"
            )

        log.info("ive.ingestion.loading", path=str(path), format=suffix)

        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".parquet", ".pq"):
            return self._load_parquet(path)
        else:
            raise ValueError(f"Unhandled suffix: {suffix}")

    def _load_csv(self, path: Path) -> "pd.DataFrame":
        """
        Load a CSV file.

        TODO:
            - Detect delimiter with csv.Sniffer
            - Handle encoding issues (try utf-8, then latin-1)
            - Strip whitespace from column names
            - Apply self.max_rows limit
        """
        import pandas as pd

        # TODO: Auto-detect delimiter and encoding
        df = pd.read_csv(path, nrows=self.max_rows)
        df.columns = df.columns.str.strip()
        log.info("ive.ingestion.loaded_csv", rows=len(df), cols=len(df.columns))
        return df

    def _load_parquet(self, path: Path) -> "pd.DataFrame":
        """
        Load a Parquet file.

        TODO:
            - Use pyarrow backend for broad compatibility
            - Apply self.max_rows limit after reading
        """
        import pandas as pd

        df = pd.read_parquet(path, engine="pyarrow")
        if self.max_rows is not None:
            df = df.head(self.max_rows)
        log.info("ive.ingestion.loaded_parquet", rows=len(df), cols=len(df.columns))
        return df
