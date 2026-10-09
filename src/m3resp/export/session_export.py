"""Session export functions."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from m3resp.export.tables import (
    events_to_rows,
    interval_data_to_rows_and_archive,
    linked_breaths_to_rows,
    parameter_results_to_rows_and_archive,
    parameters_to_rows,
    pixel_masks_to_rows_and_archive,
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
    processing_run_id: str | None = None,
) -> Path:
    """Write session summaries, result tables and array archives.

    Structured export writes session metadata and processing history, plus
    tables for populated signal, parameter, interval, mask, quality and
    linked-breath collections. Per-interval metadata is written to
    ``interval_data_metadata.json``. Array values go to
    ``parameter_result_arrays.npz``, ``interval_data_arrays.npz`` and
    ``pixel_masks.npz`` as needed, with references in their table rows.
    Exported times and units retain the values used by the session.

    Args:
        session: Session containing metadata, results, events and history.
        output_dir: Directory to create and write into. Existing files with
            the same export names are overwritten.
        summary_json: Write ``summary.json`` with metadata, quality, parameters,
            provenance and synchronization information.
        event_csvs: Write one CSV for each populated event list.
        parameters_csv: Write ``parameters.csv`` for populated parameter groups.
        postprocessing: Include ``emg_postprocessing`` in the summary and
            parameter-group table.
        structured_export: Write separate files for the scientific collections.
        processing_run_id: Run to link exported array files to when the
            session has a data-model recorder, typically
            ``WorkflowResult.processing_run_id``. ``None`` leaves archives
            without a run link.

    Returns:
        Path: The output directory used for the export.

    Raises:
        ValueError: If array values within an interval result have different shapes.
        OSError: If the output directory or files cannot be written.
        KeyError: If a supplied processing run is absent from the attached store.
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
        # Where each recording sits on the shared clock, and how it got there
        # ("manual", "none", ...). A recording missing from `sync_methods` was
        # never synchronized.
        "synchronization": {
            "start_times": _jsonable(getattr(session, "start_times", {})),
            "sync_methods": _jsonable(getattr(session, "sync_methods", {})),
        },
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
        _export_structured_collections(
            session, output_path, processing_run_id=processing_run_id
        )

    return output_path


def _export_structured_collections(
    session: Any, output_path: Path, *, processing_run_id: str | None = None
) -> None:
    """Write session metadata, history and populated scientific collections.

    Array files are linked to ``processing_run_id`` when the session has
    an attached data-model recorder.
    """

    def path_to(filename: str) -> Path:
        """Return the export path for a filename."""

        return Path(os.path.join(str(output_path), filename))

    # Convert the values per breath and the masks before writing anything, so
    # a result that cannot be stored stops the export before it has written
    # only part of the files.
    interval_data = getattr(session, "interval_data", None)
    interval_export = (
        interval_data_to_rows_and_archive(interval_data) if interval_data else None
    )
    pixel_masks = getattr(session, "pixel_masks", None)
    mask_export = pixel_masks_to_rows_and_archive(pixel_masks) if pixel_masks else None

    _write_json(path_to("session_metadata.json"), _jsonable(session.metadata))
    _write_json(
        path_to("processing_history.json"),
        {"provenance": _jsonable(session.provenance)},
    )
    if session.signals:
        _write_csv(path_to("signals_manifest.csv"), session.signals.to_manifest_rows())
    if session.parameter_results:
        rows, archive = parameter_results_to_rows_and_archive(session.parameter_results)
        _write_csv(path_to("parameter_results.csv"), rows)
        if archive:
            archive_path = path_to("parameter_result_arrays.npz")
            # numpy's stub declares an `allow_pickle: bool` keyword alongside
            # `**kwds: ArrayLike`, so mypy conservatively checks **archive's
            # value type against `bool` too; this a stub limitation, not a
            # real type error (archive is never given an `allow_pickle` key).
            np.savez_compressed(str(archive_path), **archive)  # type: ignore[arg-type]
            _link_to_run(session, archive_path, processing_run_id)
    if interval_export is not None:
        rows, archive, descriptions = interval_export
        _write_csv(path_to("interval_data.csv"), rows)
        _write_json(path_to("interval_data_metadata.json"), _jsonable(descriptions))
        if archive:
            archive_path = path_to("interval_data_arrays.npz")
            np.savez_compressed(str(archive_path), **archive)  # type: ignore[arg-type]
            _link_to_run(session, archive_path, processing_run_id)
    if mask_export is not None:
        rows, archive = mask_export
        _write_csv(path_to("pixel_masks.csv"), rows)
        archive_path = path_to("pixel_masks.npz")
        np.savez_compressed(str(archive_path), **archive)  # type: ignore[arg-type]
        _link_to_run(session, archive_path, processing_run_id)
    if session.quality:
        _write_csv(path_to("quality_flags.csv"), session.quality.to_rows())
    if session.linked_breaths:
        _write_csv(
            path_to("linked_breaths.csv"),
            linked_breaths_to_rows(session.linked_breaths),
        )


def _link_to_run(
    session: Any, archive_path: Path, processing_run_id: str | None
) -> None:
    """Link an array file to the supplied run when a recorder and run ID are given."""

    if session.datamodel is not None and processing_run_id is not None:
        session.datamodel.record_parameter_file(
            archive_path, processing_run_id=processing_run_id
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
