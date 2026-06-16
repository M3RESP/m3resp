"""Path helpers shared across the package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_optional_path(base: str | Path, value: Any) -> Path | None:
    """Resolve an optional config path relative to ``base``."""

    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.join(Path(base), path)).resolve()
