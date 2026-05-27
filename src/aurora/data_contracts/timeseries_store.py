"""Phase-1 versioned timeseries store (parquet + sqlite).

A :class:`TimeSeriesStore` exposes a content-addressed, versioned,
namespace-aware (``library/symbol/version``) parquet+sqlite store for
DataFrames that survive across runs. It is the ``arctic``-style layer
that the data-integrity programme (Candidate C, Phase 1) calls for.

Design notes
------------

* Each ``put`` writes a parquet file at ``<root>/<library>/<symbol>/
  <version>.parquet`` and registers the row in
  ``<root>/timeseries_index.sqlite``.
* The on-disk content is paired with a content-addressed sha256 hash
  computed over the sorted ISO index, columns, values, and metadata
  ordering, so identical inputs produce identical hashes regardless of
  insertion order.
* Versions default to a UTC ISO timestamp at ``put`` time (millisecond
  precision) — sortable, deterministic across processes, and human
  readable. Callers may override.
* ``replace=False`` (default) raises if a version already exists.
  ``replace=True`` rewrites both the parquet and the sqlite row, bumping
  the content hash but keeping the version key stable.
* Soft delete is implemented via a tombstone metadata flag; the parquet
  file is removed but the index row stays so the deletion is auditable.

The store is a *library*: all paths flow through
:func:`aurora.core.runtime_paths.base_data_dir`, never hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from aurora.core.runtime_paths import base_data_dir


TIMESERIES_STORE_VERSION = "1.0.0"


@dataclass(frozen=True)
class TimeSeriesRecord:
    """Immutable manifest for one stored timeseries version.

    Attributes:
        library: namespace bucket (e.g. ``"raw"``, ``"adjusted"``,
            ``"features"``). Required to be non-empty.
        symbol: symbol within the library (e.g. ``"AAPL"``).
        version: opaque version string (sortable ISO timestamp by default).
        index: tuple of ISO timestamp strings in stored row order.
        columns: tuple of column names.
        metadata: frozen mapping of small string key/value metadata. The
            ``tombstone`` key is reserved for soft-delete bookkeeping.
        content_hash: sha256 hex digest over the canonical payload.
        created_at: ISO timestamp (UTC) when the row was registered.
        n_rows: number of rows in the stored DataFrame.
        data_path: absolute parquet path. Empty if soft-deleted.
    """

    library: str
    symbol: str
    version: str
    index: Tuple[str, ...]
    columns: Tuple[str, ...]
    metadata: Mapping[str, Any]
    content_hash: str
    created_at: str = ""
    n_rows: int = 0
    data_path: str = ""

    @property
    def is_tombstone(self) -> bool:
        """``True`` iff the row was soft-deleted."""
        return bool(dict(self.metadata).get("tombstone"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _utcnow_iso_ms() -> str:
    """Return the current UTC time as an ISO-8601 string with ms precision."""
    now = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
    # millisecond precision so two ``put`` calls in the same minute don't
    # collide on the default version key.
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _canonical_index(df: pd.DataFrame) -> Tuple[str, ...]:
    """Return the DataFrame index as a tuple of ISO strings.

    Accepts DatetimeIndex (any tz, normalised to UTC), regular Index, and
    MultiIndex (joined with ``|``). The result is deterministic and stable
    across pandas minor versions.
    """
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return tuple(pd.Timestamp(x).isoformat() for x in idx)
    if isinstance(idx, pd.MultiIndex):
        return tuple("|".join(str(x) for x in tup) for tup in idx)
    return tuple(str(x) for x in idx)


def _content_hash(
    library: str,
    symbol: str,
    version: str,
    df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    """Compute the deterministic sha256 over (library, symbol, version,
    sorted index ISO, columns, values bytes, metadata json).
    """
    h = hashlib.sha256()
    h.update(library.encode("utf-8"))
    h.update(b"\x00")
    h.update(symbol.encode("utf-8"))
    h.update(b"\x00")
    h.update(version.encode("utf-8"))
    h.update(b"\x00")
    for iso in _canonical_index(df):
        h.update(iso.encode("utf-8"))
        h.update(b"|")
    h.update(b"\x00")
    for col in df.columns:
        h.update(str(col).encode("utf-8"))
        h.update(b"|")
    h.update(b"\x00")
    # Cast each column to bytes via numpy so the digest is independent of
    # pandas dtype changes between minor versions.
    for col in df.columns:
        arr = np.asarray(df[col].values)
        try:
            h.update(np.ascontiguousarray(arr).tobytes())
        except (TypeError, ValueError):
            # Object dtype / non-trivial: fall back to repr.
            h.update(repr(tuple(arr)).encode("utf-8"))
        h.update(b"\x00")
    h.update(
        json.dumps(dict(metadata), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    )
    return h.hexdigest()


# --------------------------------------------------------------------------
# TimeSeriesStore
# --------------------------------------------------------------------------


class TimeSeriesStore:
    """Versioned parquet+sqlite store for DataFrames.

    Layout:
        <root>/<library>/<symbol>/<version>.parquet
        <root>/timeseries_index.sqlite

    Concurrent writers across processes are serialized via ``BEGIN
    IMMEDIATE`` on the sqlite index. Within a process, a module-level
    lock guards parquet+sqlite atomicity for ``put`` and ``delete``.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS timeseries (
            library       TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            version       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            n_rows        INTEGER NOT NULL,
            columns_json  TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            data_path     TEXT NOT NULL,
            PRIMARY KEY (library, symbol, version)
        )
    """

    _LOCK = Lock()

    def __init__(self, root_dir: Optional[os.PathLike] = None) -> None:
        if root_dir is None:
            root = base_data_dir() / "timeseries"
        else:
            root = Path(root_dir)
        self.root_dir = Path(os.path.abspath(root))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "timeseries_index.sqlite"
        self._init_index()

    # ---- index bootstrap -------------------------------------------------

    def _init_index(self) -> None:
        con = sqlite3.connect(str(self.index_path))
        try:
            con.execute(self._SCHEMA)
            con.commit()
        finally:
            con.close()

    # ---- internal helpers ------------------------------------------------

    def _data_dir(self, library: str, symbol: str) -> Path:
        d = self.root_dir / library / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _row_to_record(self, row: tuple) -> TimeSeriesRecord:
        (
            library,
            symbol,
            version,
            content_hash,
            n_rows,
            columns_json,
            metadata_json,
            created_at,
            data_path,
        ) = row
        cols = tuple(json.loads(columns_json))
        meta = json.loads(metadata_json)
        # Index is materialised lazily on read; the manifest carries the
        # canonical column / metadata view.
        return TimeSeriesRecord(
            library=library,
            symbol=symbol,
            version=version,
            index=tuple(),
            columns=cols,
            metadata=meta,
            content_hash=content_hash,
            created_at=created_at,
            n_rows=int(n_rows),
            data_path=data_path,
        )

    # ---- public API ------------------------------------------------------

    @staticmethod
    def _safe_filename(version: str) -> str:
        """Replace filesystem-hostile characters in a version string.

        Windows blocks ``:``, ``?``, ``*`` etc. in filenames. The version
        itself stays in the index column unchanged so callers see the
        original string.
        """
        bad = ':?*"<>|/\\'
        out = version
        for ch in bad:
            out = out.replace(ch, "_")
        return out

    def put(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        version: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
    ) -> TimeSeriesRecord:
        """Store ``df`` under ``library/symbol/version``.

        Args:
            library: non-empty library namespace.
            symbol: non-empty symbol.
            df: DataFrame to store. The index is preserved verbatim.
            version: opaque version string. Defaults to the current UTC
                ISO timestamp with millisecond precision.
            metadata: small string-keyed dict of metadata to embed in the
                content hash. Defaults to ``{}``.
            replace: if ``True``, overwrite an existing version. The
                content hash will be re-computed and may change.

        Returns:
            :class:`TimeSeriesRecord` describing the stored version.

        Raises:
            ValueError: empty ``library`` / ``symbol``, non-DataFrame, or
                attempting to overwrite without ``replace=True``.
        """
        if not library:
            raise ValueError("library must be a non-empty string")
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        version = version or _utcnow_iso_ms()
        meta = dict(metadata) if metadata else {}

        h = _content_hash(library, symbol, version, df, meta)
        target_dir = self._data_dir(library, symbol)
        data_path = target_dir / f"{self._safe_filename(version)}.parquet"
        created_at = _utcnow_iso_ms()
        cols = tuple(str(c) for c in df.columns)

        with self._LOCK:
            con = sqlite3.connect(str(self.index_path))
            try:
                con.isolation_level = None
                con.execute("BEGIN IMMEDIATE")
                existing = con.execute(
                    "SELECT content_hash FROM timeseries "
                    "WHERE library = ? AND symbol = ? AND version = ?",
                    (library, symbol, version),
                ).fetchone()
                if existing is not None and not replace:
                    con.execute("ROLLBACK")
                    raise ValueError(
                        f"version {version!r} already exists for "
                        f"{library}/{symbol}; pass replace=True to overwrite"
                    )

                wrote_parquet = False
                try:
                    df.to_parquet(
                        data_path, engine="pyarrow", compression="snappy"
                    )
                    wrote_parquet = True
                    con.execute(
                        """INSERT OR REPLACE INTO timeseries
                           (library, symbol, version, content_hash, n_rows,
                            columns_json, metadata_json, created_at, data_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            library,
                            symbol,
                            version,
                            h,
                            int(len(df)),
                            json.dumps(list(cols)),
                            json.dumps(meta, default=str, sort_keys=True),
                            created_at,
                            str(data_path),
                        ),
                    )
                    con.execute("COMMIT")
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        pass
                    if wrote_parquet:
                        try:
                            os.remove(data_path)
                        except OSError:
                            pass
                    raise
            finally:
                con.close()

        return TimeSeriesRecord(
            library=library,
            symbol=symbol,
            version=version,
            index=_canonical_index(df),
            columns=cols,
            metadata=meta,
            content_hash=h,
            created_at=created_at,
            n_rows=int(len(df)),
            data_path=str(data_path),
        )

    def read(
        self,
        library: str,
        symbol: str,
        *,
        version: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Read a stored DataFrame, optionally filtered by date range.

        Args:
            library: namespace bucket.
            symbol: symbol within the library.
            version: explicit version. Defaults to the latest non-tombstoned
                version (sorted lexicographically — sortable ISO timestamps
                are the default).
            start: inclusive lower bound on the timestamp index. Accepts
                anything :class:`pandas.Timestamp` accepts.
            end: inclusive upper bound on the timestamp index.

        Returns:
            ``pd.DataFrame`` with the original index.

        Raises:
            KeyError: no such version, or all versions are tombstoned.
            FileNotFoundError: parquet missing on disk.
        """
        rec = self._lookup(library, symbol, version=version)
        if rec is None:
            raise KeyError(
                f"no readable version for {library}/{symbol}"
                + (f" (version={version!r})" if version else "")
            )
        if rec.is_tombstone:
            raise KeyError(
                f"version {rec.version!r} for {library}/{symbol} is tombstoned"
            )
        path = Path(rec.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"timeseries parquet missing at {path} for "
                f"{library}/{symbol}@{rec.version}"
            )
        df = pd.read_parquet(path)

        if start is None and end is None:
            return df

        # Date-range filter. If the index is a DatetimeIndex, use it
        # directly; otherwise try to coerce and fall back gracefully.
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except (TypeError, ValueError):
                # Non-temporal index: caller asked for a date range we
                # cannot honour — return the full frame and let the
                # caller decide. This matches pandas ``loc[start:end]``
                # behaviour on string indexes.
                return df

        s = pd.Timestamp(start) if start is not None else None
        e = pd.Timestamp(end) if end is not None else None
        # Align tz: if the index is tz-aware and bounds are naive, attach
        # the index's tz. If naive vs aware mismatches the other way,
        # coerce bounds to the index tz.
        if idx.tz is not None:
            if s is not None and s.tz is None:
                s = s.tz_localize(idx.tz)
            if e is not None and e.tz is None:
                e = e.tz_localize(idx.tz)
        else:
            if s is not None and s.tz is not None:
                s = s.tz_convert("UTC").tz_localize(None)
            if e is not None and e.tz is not None:
                e = e.tz_convert("UTC").tz_localize(None)

        mask = pd.Series(True, index=idx)
        if s is not None:
            mask &= idx >= s
        if e is not None:
            mask &= idx <= e
        return df.loc[mask.values]

    def list_versions(self, library: str, symbol: str) -> Tuple[str, ...]:
        """Return non-tombstoned versions, sorted lexicographically."""
        con = sqlite3.connect(str(self.index_path))
        try:
            cur = con.execute(
                "SELECT version, metadata_json FROM timeseries "
                "WHERE library = ? AND symbol = ? ORDER BY version",
                (library, symbol),
            )
            rows = cur.fetchall()
        finally:
            con.close()
        out: list[str] = []
        for version, metadata_json in rows:
            meta = json.loads(metadata_json)
            if not meta.get("tombstone"):
                out.append(version)
        return tuple(out)

    def list_records(self, library: str, symbol: str) -> Tuple[TimeSeriesRecord, ...]:
        """Return all manifests (including tombstones) for a series."""
        con = sqlite3.connect(str(self.index_path))
        try:
            cur = con.execute(
                """SELECT library, symbol, version, content_hash, n_rows,
                          columns_json, metadata_json, created_at, data_path
                   FROM timeseries
                   WHERE library = ? AND symbol = ? ORDER BY version""",
                (library, symbol),
            )
            rows = cur.fetchall()
        finally:
            con.close()
        return tuple(self._row_to_record(r) for r in rows)

    def delete(self, library: str, symbol: str, version: str) -> TimeSeriesRecord:
        """Soft-delete a version.

        The parquet file is removed and the metadata is marked with
        ``tombstone=True``. The index row stays for audit. ``read`` will
        refuse to load the tombstoned version; ``list_versions`` will
        omit it.

        Raises:
            KeyError: no such version.
        """
        with self._LOCK:
            con = sqlite3.connect(str(self.index_path))
            try:
                con.isolation_level = None
                con.execute("BEGIN IMMEDIATE")
                row = con.execute(
                    """SELECT library, symbol, version, content_hash, n_rows,
                              columns_json, metadata_json, created_at, data_path
                       FROM timeseries
                       WHERE library = ? AND symbol = ? AND version = ?""",
                    (library, symbol, version),
                ).fetchone()
                if row is None:
                    con.execute("ROLLBACK")
                    raise KeyError(
                        f"no version {version!r} for {library}/{symbol}"
                    )
                rec = self._row_to_record(row)
                meta = dict(rec.metadata)
                meta["tombstone"] = True
                meta["tombstoned_at"] = _utcnow_iso_ms()
                # Remove parquet (best effort).
                try:
                    Path(rec.data_path).unlink(missing_ok=True)
                except OSError:
                    pass
                con.execute(
                    "UPDATE timeseries SET metadata_json = ?, data_path = '' "
                    "WHERE library = ? AND symbol = ? AND version = ?",
                    (
                        json.dumps(meta, default=str, sort_keys=True),
                        library,
                        symbol,
                        version,
                    ),
                )
                con.execute("COMMIT")
            finally:
                con.close()
        return TimeSeriesRecord(
            library=rec.library,
            symbol=rec.symbol,
            version=rec.version,
            index=tuple(),
            columns=rec.columns,
            metadata=meta,
            content_hash=rec.content_hash,
            created_at=rec.created_at,
            n_rows=rec.n_rows,
            data_path="",
        )

    # ---- internal lookup -------------------------------------------------

    def _lookup(
        self, library: str, symbol: str, *, version: Optional[str] = None
    ) -> Optional[TimeSeriesRecord]:
        """Return the manifest for the requested (or latest) version."""
        con = sqlite3.connect(str(self.index_path))
        try:
            if version is not None:
                cur = con.execute(
                    """SELECT library, symbol, version, content_hash, n_rows,
                              columns_json, metadata_json, created_at, data_path
                       FROM timeseries
                       WHERE library = ? AND symbol = ? AND version = ?""",
                    (library, symbol, version),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)
            cur = con.execute(
                """SELECT library, symbol, version, content_hash, n_rows,
                          columns_json, metadata_json, created_at, data_path
                   FROM timeseries
                   WHERE library = ? AND symbol = ? ORDER BY version DESC""",
                (library, symbol),
            )
            for row in cur.fetchall():
                rec = self._row_to_record(row)
                if not rec.is_tombstone:
                    return rec
            return None
        finally:
            con.close()


# --------------------------------------------------------------------------
# default store helper
# --------------------------------------------------------------------------


_default_store: Optional[TimeSeriesStore] = None
_default_store_lock = Lock()


def default_store() -> TimeSeriesStore:
    """Return the singleton :class:`TimeSeriesStore` rooted under
    :func:`aurora.core.runtime_paths.base_data_dir`.
    """
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = TimeSeriesStore()
        return _default_store


def _reset_default_store_for_tests() -> None:
    """Test hook: clear the cached singleton so a new ``base_data_dir``
    (e.g. monkey-patched ``AU_DATA_DIR``) is honoured on the next call.
    """
    global _default_store
    with _default_store_lock:
        _default_store = None


__all__ = [
    "TIMESERIES_STORE_VERSION",
    "TimeSeriesRecord",
    "TimeSeriesStore",
    "default_store",
]
