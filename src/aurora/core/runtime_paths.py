"""Centralized runtime path resolution.

All persistent Aurora runtime artifacts (caches, snapshots, audit logs,
research archives, OOS locks) flow through this module. Paths are configurable
via environment variables and default to platformdirs locations isolated from
the package directory.

Env vars (override defaults). Both the canonical ``AU_*`` names (Aurora
1.5+) and the legacy ``QF_*`` names (read with a ``DeprecationWarning``
during the shim window per R23 / R76) are accepted::

    AU_DATA_DIR / QF_DATA_DIR              base data dir
    AU_CACHE_DIR / QF_CACHE_DIR            price/data cache dir
    QF_CACHE                               legacy alias for $AU_CACHE_DIR
    AU_SNAPSHOT_ROOT / QF_SNAPSHOT_ROOT    SnapshotStore root_dir
    AU_AUDIT_LOG / QF_AUDIT_LOG            audit trail JSONL
    AU_GATEWAY_AUDIT / QF_GATEWAY_AUDIT    agent gateway audit chain
    AU_OOS_LOCK / QF_OOS_LOCK              OOSGuard lock file
    AU_RESEARCH_ARCHIVE / QF_RESEARCH_ARCHIVE   research factory archive
    AU_REVIEW_QUEUE / QF_REVIEW_QUEUE      research factory review queue
    AU_CONFIG_DIR / QF_CONFIG_DIR          user-overridable config dir

These paths NEVER point inside the installed package (site-packages stays
read-only when installed from wheel).
"""
from __future__ import annotations

import os
from pathlib import Path

from aurora.core.env_compat import aurora_env


def _platformdirs_base() -> Path:
    """Default user data dir via platformdirs."""
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("aurora", appauthor=False))
    except ImportError:
        return Path(os.path.expanduser("~")) / ".aurora"


def base_data_dir() -> Path:
    """Base dir for all runtime artifacts. Override via $AU_DATA_DIR / $QF_DATA_DIR."""
    raw = aurora_env("AU_DATA_DIR", "QF_DATA_DIR")
    p = Path(raw) if raw else _platformdirs_base()
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """Price/data cache dir. Override via $AU_CACHE_DIR / $QF_CACHE_DIR / legacy $QF_CACHE."""
    raw = (
        aurora_env("AU_CACHE_DIR", "QF_CACHE_DIR")
        or aurora_env("AU_CACHE", "QF_CACHE")
    )
    p = Path(raw) if raw else base_data_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def snapshot_root() -> Path:
    """SnapshotStore root_dir. Override via $AU_SNAPSHOT_ROOT / $QF_SNAPSHOT_ROOT.

    Contains parquet files + snapshots_index.sqlite. NOT just a single DB file.
    """
    raw = aurora_env("AU_SNAPSHOT_ROOT", "QF_SNAPSHOT_ROOT")
    p = Path(raw) if raw else base_data_dir() / "snapshots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def audit_log_path() -> Path:
    """SOC2 audit trail JSONL. Override via $AU_AUDIT_LOG / $QF_AUDIT_LOG."""
    raw = aurora_env("AU_AUDIT_LOG", "QF_AUDIT_LOG")
    return Path(raw) if raw else base_data_dir() / "audit_trail.jsonl"


def gateway_audit_path() -> Path:
    """Agent gateway hash-chained audit JSONL. Override via $AU_GATEWAY_AUDIT / $QF_GATEWAY_AUDIT."""
    raw = aurora_env("AU_GATEWAY_AUDIT", "QF_GATEWAY_AUDIT")
    return Path(raw) if raw else base_data_dir() / "gateway_audit.jsonl"


def oos_lock_path() -> Path:
    """OOSGuard cross-process lock file. Override via $AU_OOS_LOCK / $QF_OOS_LOCK."""
    raw = aurora_env("AU_OOS_LOCK", "QF_OOS_LOCK")
    return Path(raw) if raw else base_data_dir() / ".oos_lock.json"


def research_archive_path() -> Path:
    """ResearchFactory rejection archive JSONL. Override via $AU_RESEARCH_ARCHIVE / $QF_RESEARCH_ARCHIVE."""
    raw = aurora_env("AU_RESEARCH_ARCHIVE", "QF_RESEARCH_ARCHIVE")
    return Path(raw) if raw else base_data_dir() / "research_archive.jsonl"


def review_queue_path() -> Path:
    """ResearchFactory review queue JSONL. Override via $AU_REVIEW_QUEUE / $QF_REVIEW_QUEUE."""
    raw = aurora_env("AU_REVIEW_QUEUE", "QF_REVIEW_QUEUE")
    return Path(raw) if raw else base_data_dir() / "research_review_queue.jsonl"


def user_config_dir() -> Path:
    """User-overridable config dir. Override via $AU_CONFIG_DIR / $QF_CONFIG_DIR.

    For built-in package configs (e.g. protocol_policy.yaml), use
    importlib.resources to read from the installed package directly.
    """
    raw = aurora_env("AU_CONFIG_DIR", "QF_CONFIG_DIR")
    p = Path(raw) if raw else base_data_dir() / "config"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Eager-evaluated module constants for backwards-compat with code that imports
# them as values rather than calling the functions. Tests using monkeypatch on
# these constants should switch to setenv on the corresponding $AU_* / $QF_*
# vars.
__all__ = [
    "base_data_dir",
    "cache_dir",
    "snapshot_root",
    "audit_log_path",
    "gateway_audit_path",
    "oos_lock_path",
    "research_archive_path",
    "review_queue_path",
    "user_config_dir",
]
