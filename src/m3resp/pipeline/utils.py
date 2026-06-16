"""Utility helpers used by pipeline steps and the engine."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger


def slice_by_index(data: Any, *, start: int, end: int) -> Any:
    return data[slice(start, end)]


def slice_by_time(data: Any, *, start: float, end: float) -> Any:
    return data.t[slice(start, end)]


def slice_signal_by_mode(
    data: Any,
    *,
    start: int | float,
    end: int | float,
    slicing_mode: str,
) -> Any:
    """Slice a signal by sample index or time selector."""

    if slicing_mode == "index":
        return slice_by_index(data, start=int(start), end=int(end))
    if slicing_mode == "time":
        return slice_by_time(data, start=float(start), end=float(end))
    raise ValueError("slicing_mode must be 'index' or 'time'.")


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
) -> None:
    """Log a workflow summary with readable colored keys."""

    logger.opt(colors=True).success("<bold>{}</bold>", title)
    logger.opt(colors=True).info("<cyan>Output directory</cyan>: {}", output_dir)
    for key, value in summary.items():
        if _log_postprocessing_summary(key, value):
            continue
        logger.opt(colors=True).info("<cyan>{}</cyan>: {}", key, value)


def _log_postprocessing_summary(key: str, value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False

    label = key.replace("_", " ")
    if key.endswith("postprocessing_available"):
        available = ", ".join(
            f"{cat.replace('_', ' ')} ({count})" for cat, count in value.items()
        )
        logger.opt(colors=True).info("<cyan>{}</cyan>: {}", label, available or "none")
        return True

    if key.endswith("postprocessing_computed"):
        logger.opt(colors=True).info("<cyan>{}</cyan>:", label)
        if not value:
            logger.info("  none")
            return True
        for cat, functions in value.items():
            logger.opt(colors=True).info(
                "  <blue>{}</blue>: {}",
                cat.replace("_", " "),
                ", ".join(str(f) for f in functions) if functions else "none",
            )
        return True

    if key.endswith("postprocessing_skipped"):
        logger.opt(colors=True).info("<cyan>{}</cyan>:", label)
        if not value:
            logger.info("  none")
            return True
        for cat, reason in value.items():
            logger.opt(colors=True).info(
                "  <blue>{}</blue>: {}", cat.replace("_", " "), reason
            )
        return True

    return False
