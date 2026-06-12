"""Convenience helpers for configured workflow scripts."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from m3resp.core.config import WorkflowConfig, load_workflow_config


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


def configure_workflow_paths(*sibling_repositories: str) -> Path:
    """Add local source paths used by workflow scripts and return the repo root."""

    repo_root = find_repo_root()
    paths = [Path(os.path.join(str(repo_root), "src"))]
    paths.extend(
        Path(os.path.join(str(repo_root.parent), name)) for name in sibling_repositories
    )

    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return repo_root


def configure_workflow_logging() -> None:
    """Configure colorful Loguru output for runnable workflow scripts."""

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def get_config_path(default_config_path: str) -> str:
    """Ask for a workflow config path, defaulting to the provided path."""

    config_path = input(
        f"Enter a config path [press enter for default:{default_config_path}]: "
    ).strip()
    return os.path.normpath(config_path) if config_path else default_config_path


def load_workflow_config_with_raw(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
) -> tuple[WorkflowConfig, dict[str, Any]]:
    """Load typed workflow config and return the raw YAML mapping as well."""

    if isinstance(config, WorkflowConfig):
        return config, {}

    import yaml

    config_path = Path(config).expanduser().resolve()
    cfg = load_workflow_config(config_path, root=root)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return cfg, dict(raw)


def coerce_slice_value(
    override_value: int | float | None,
    config_value: Any,
    slicing_mode: str,
    *,
    value_name: str = "slice value",
) -> int | float:
    """Resolve and coerce an index/time slice boundary from config and overrides."""

    value = override_value if override_value is not None else config_value
    if value is None:
        raise ValueError(f"Workflow requires {value_name}.")
    if slicing_mode == "index":
        return int(value)
    if slicing_mode == "time":
        return float(value)
    raise ValueError("slicing_mode must be 'index' or 'time'.")


def slice_by_index(data: Any, *, start: int, end: int) -> Any:
    """Slice an indexable data object by sample index."""

    return data[slice(start, end)]


def slice_by_time(data: Any, *, start: float, end: float) -> Any:
    """Slice a data object by time using its ``.t`` selector."""

    return data.t[slice(start, end)]


def subject_result_filename(
    subject_id: str,
    mode: str,
    timepoint: Any,
    selection: str,
) -> str:
    """Return the conventional subject result filename."""

    timepoint_part = f"{timepoint}-" if timepoint else ""
    return f"{subject_id}-{mode}-{timepoint_part}{selection}.txt"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a dictionary-like payload as deterministic JSON."""

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def log_workflow_summary(
    title: str,
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    active_modules: Mapping[str, bool] | None = None,
) -> None:
    """Log a workflow summary with readable colored keys."""

    logger.opt(colors=True).success("<bold>{}</bold>", title)
    logger.opt(colors=True).info("<cyan>Output directory</cyan>: {}", output_dir)

    if active_modules is not None:
        modules = ", ".join(
            f"{name}={'on' if enabled else 'off'}"
            for name, enabled in active_modules.items()
        )
        logger.opt(colors=True).info("<cyan>Active modules</cyan>: {}", modules)

    for key, value in summary.items():
        if _log_postprocessing_summary(key, value):
            continue
        logger.opt(colors=True).info("<cyan>{}</cyan>: {}", key, value)


def _log_postprocessing_summary(key: str, value: Any) -> bool:
    """Log EMG postprocessing dictionaries as readable grouped sections."""

    if not isinstance(value, Mapping):
        return False

    label = _format_summary_label(key)
    if key.endswith("postprocessing_available"):
        available = ", ".join(
            f"{category.replace('_', ' ')} ({count})"
            for category, count in value.items()
        )
        logger.opt(colors=True).info(
            "<cyan>{}</cyan>: {}",
            label,
            available or "none",
        )
        return True

    if key.endswith("postprocessing_computed"):
        logger.opt(colors=True).info("<cyan>{}</cyan>:", label)
        if not value:
            logger.info("  none")
            return True
        for category, functions in value.items():
            logger.opt(colors=True).info(
                "  <blue>{}</blue>: {}",
                category.replace("_", " "),
                _format_function_list(functions),
            )
        return True

    if key.endswith("postprocessing_skipped"):
        logger.opt(colors=True).info("<cyan>{}</cyan>:", label)
        if not value:
            logger.info("  none")
            return True
        for category, reason in value.items():
            logger.opt(colors=True).info(
                "  <blue>{}</blue>: {}",
                category.replace("_", " "),
                reason,
            )
        return True

    return False


def _format_summary_label(key: str) -> str:
    """Return a readable label for a summary key."""

    if key.startswith("emg_"):
        return f"EMG {key.removeprefix('emg_').replace('_', ' ')}"
    return key.replace("_", " ")


def _format_function_list(value: Any) -> str:
    """Return a readable function list for log output."""

    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)
