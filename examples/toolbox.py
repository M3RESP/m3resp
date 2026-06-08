"""Small helpers shared by runnable example scripts."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # PyYAML (already a transitive dependency of many scientific stacks)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleFlags:
    """On/off switches for each modality."""

    eit: bool = True
    emg: bool = True
    vent: bool = True


@dataclass(frozen=True)
class EITConfig:
    """EIT-specific settings."""

    file: Path | None = None
    vendor: str = "draeger"


@dataclass(frozen=True)
class EMGConfig:
    """EMG-specific settings."""

    file: Path | None = None


@dataclass(frozen=True)
class VentConfig:
    """Ventilator-specific settings."""

    file: Path | None = None


@dataclass(frozen=True)
class AlignmentConfig:
    """Inter-modality alignment settings."""

    method: str = "manual_offset"
    manual_offset_seconds: float = 0.0


@dataclass(frozen=True)
class OutputConfig:
    """Resolved output directories."""

    combined: Path = Path("output/multimodal-summary")
    eit_only: Path = Path("output/eit-summary")
    emg_only: Path = Path("output/emg-summary")


@dataclass(frozen=True)
class ExampleConfig:
    """Typed, immutable container for everything in ``config.yaml``."""

    modules: ModuleFlags
    eit: EITConfig
    emg: EMGConfig
    vent: VentConfig
    alignment: AlignmentConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def find_repo_root(start: Path | None = None) -> Path:
    """Find the local m3resp repository root."""

    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (
            Path(os.path.join(str(candidate), "pyproject.toml")).exists()
            and Path(os.path.join(str(candidate), "src", "m3resp")).exists()
        ):
            return candidate
    raise RuntimeError("Could not find the m3resp repository root.")


def configure_example_paths(*sibling_repositories: str) -> Path:
    """Add local source paths used by examples and return the repo root."""

    repo_root = find_repo_root()
    paths = [Path(os.path.join(str(repo_root), "src"))]
    paths.extend(
        Path(os.path.join(str(repo_root.parent), name)) for name in sibling_repositories
    )

    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return repo_root


def save_figures(output_dir: Path, figures: Mapping[str, Any], dpi: int = 150) -> None:
    """Save named matplotlib figures into an output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, figure in figures.items():
        figure.savefig(Path(os.path.join(str(output_dir), filename)), dpi=dpi)


def _resolve(root: Path, relative: str) -> Path:
    """Join *relative* onto *root* and return as an absolute ``Path``."""

    return Path(os.path.join(str(root), relative))


def load_config(repo_root: Path | None = None) -> ExampleConfig:
    """Load ``examples/config.yaml`` and return a typed :class:`ExampleConfig`.

    All relative file-paths in the YAML are resolved against *repo_root*.

    Parameters
    ----------
    repo_root:
        Repository root returned by :func:`configure_example_paths`.
        When *None*, :func:`find_repo_root` is called automatically.

    Returns
    -------
    ExampleConfig
        Immutable, fully-resolved configuration object.
    """

    root = repo_root or find_repo_root()
    config_path = Path(os.path.join(str(root), "examples", "config.yaml"))

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    modules = ModuleFlags(**raw.get("modules", {}))

    eit_raw = raw.get("eit", {})
    emg_raw = raw.get("emg", {})
    vent_raw = raw.get("vent", {})

    return ExampleConfig(
        modules=modules,
        eit=EITConfig(
            file=_resolve(root, eit_raw["file"]) if "file" in eit_raw else None,
            vendor=eit_raw.get("vendor", "draeger"),
        ),
        emg=EMGConfig(
            file=_resolve(root, emg_raw["file"]) if "file" in emg_raw else None,
        ),
        vent=VentConfig(
            file=_resolve(root, vent_raw["file"]) if "file" in vent_raw else None,
        ),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
        output=OutputConfig(
            **{key: _resolve(root, val) for key, val in raw.get("output", {}).items()}
        ),
    )
