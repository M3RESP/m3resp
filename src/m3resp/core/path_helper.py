"""Path helpers shared by configuration and workflow utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def infer_repo_root(start: str | Path) -> Path:
    """Return the nearest m3resp repository root at or above ``start``."""

    path = Path(start).expanduser().resolve()
    current = path if path.is_dir() else path.parent
    for candidate in [current, *current.parents]:
        if (
            Path(os.path.join(candidate, "pyproject.toml")).exists()
            and Path(os.path.join(candidate, "src", "m3resp")).exists()
        ):
            return candidate
    raise RuntimeError("Could not find the m3resp repository root.")


def infer_config_root(config_path: str | Path) -> Path:
    """Infer the path base used to resolve workflow config paths."""

    path = Path(config_path).expanduser().resolve()
    try:
        return infer_repo_root(path)
    except RuntimeError:
        return path.parent


def resolve_optional_path(base: str | Path, value: Any) -> Path | None:
    """Resolve an optional config path relative to ``base``."""

    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.join(Path(base), path)).resolve()
