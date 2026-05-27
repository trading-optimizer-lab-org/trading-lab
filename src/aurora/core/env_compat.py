from __future__ import annotations

import os


def aurora_env(primary: str, legacy: str | None = None) -> str | None:
    return os.environ.get(primary) or (os.environ.get(legacy) if legacy else None)

