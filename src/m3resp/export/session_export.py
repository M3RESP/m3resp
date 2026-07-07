"""Session export functions."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from m3resp.export.tables import (
    events_to_rows,
    linked_breaths_to_rows,
    parameters_to_rows,
)


def export_session_summary(
    session: Any,
    output_dir: str | Path,
    *,
    summary_json: bool = True,
    event_csvs: bool = True,
    parameters_csv: bool = True,
    postprocessing: bool = True,
    structured_export: bool = True,
) -> Path:
    """Export a minimal CSV/JSON summary for an M3Resp session.

    ``structured_export`` (Milestone 2.6, plan_stage2.md Sec 22) additionally
    writes the Layer 1 typed collections (Milestone 2.1/2.2/2.5) each to their
    own file: ``session_metadata.json``, ``signals_manifest.csv``,
    ``parameter_results.csv``, ``quality_flags.csv``, ``linked_breaths.csv``,
    and ``processing_history.json``. These are additive - ``summary.json``
    and the per-event-list CSVs above are unchanged and keep their Stage 1
    shape.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parameters = dict(session.parameters)
    if not postprocessing:
        parameters.pop("emg_postprocessing", None)

    summary = {
        "metadata": _jsonable(session.metadata),
        "quality": _jsonable(session.quality),
        "parameters": _jsonable(parameters),
        "provenance": _jsonable(session.provenance),
    }

    if summary_json:
        Path(os.path.join(output_path, "summary.json")).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    if event_csvs:
        for name, events in session.events.items():
            if events:
                _write_csv(
                    Path(os.path.join(output_path, f"{name}.csv")),
                    events_to_rows(events),
                )

    if parameters_csv and parameters:
        _write_csv(
            Path(os.path.join(output_path, "parameters.csv")),
            parameters_to_rows(parameters),
        )

    if structured_export:
        _export_structured_collections(session, output_path)

    return output_path


def _export_structured_collections(session: Any, output_path: Path) -> None:
    """Write the Milestone 2.6 per-entity files (see ``export_session_summary``)."""

    _write_json(output_path / "session_metadata.json", _jsonable(session.metadata))
    _write_json(
        output_path / "processing_history.json",
        {"provenance": _jsonable(session.provenance)},
    )
    if session.signals:
        _write_csv(
            output_path / "signals_manifest.csv", session.signals.to_manifest_rows()
        )
    if session.parameter_results:
        _write_csv(
            output_path / "parameter_results.csv", session.parameter_results.to_rows()
        )
    if session.quality:
        _write_csv(output_path / "quality_flags.csv", session.quality.to_rows())
    if session.linked_breaths:
        _write_csv(
            output_path / "linked_breaths.csv",
            linked_breaths_to_rows(session.linked_breaths),
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if _is_dataclass_instance(value):
        return _jsonable(asdict(cast("DataclassInstance", value)))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)
