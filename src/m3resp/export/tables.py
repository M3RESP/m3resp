"""Convert M3Resp objects into table rows."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.data.events import BreathEvent, Event, event_to_dict

if TYPE_CHECKING:
    from m3resp.data.event_data import IntervalData
    from m3resp.data.linked_breath import LinkedBreath
    from m3resp.data.parameters import ParameterResult
    from m3resp.data.pixel_maps import PixelMask

#: `LinkedBreath` modality slot -> attribute name, in row-column order.
_LINKED_BREATH_SLOTS = ("eit", "emg", "ventilator")


def events_to_rows(events: list[Event] | list[BreathEvent]) -> list[dict[str, Any]]:
    """Convert event dataclasses to serializable rows."""

    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(event_to_dict(event))
    return rows


def parameters_to_rows(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten top-level parameter groups into serializable rows."""

    rows: list[dict[str, Any]] = []
    for modality, values in parameters.items():
        if isinstance(values, dict):
            for name, value in values.items():
                rows.append({"modality": modality, "name": name, "value": value})
        else:
            rows.append({"modality": modality, "name": "value", "value": values})
    return rows


def linked_breaths_to_rows(linked_breaths: list[LinkedBreath]) -> list[dict[str, Any]]:
    """Flatten `LinkedBreath` objects into one row per link (Milestone 2.5/2.6).

    Each modality's breath fields are prefixed (``eit_start_time``,
    ``emg_extremum_time``, ``emg_extremum_index``, ...) and left ``None`` when that modality has no
    breath in the link, so the CSV has a stable column set regardless of
    which modalities matched.
    """

    rows: list[dict[str, Any]] = []
    for linked in linked_breaths:
        row: dict[str, Any] = {
            "modalities": "+".join(linked.modalities),
            "confidence": linked.confidence,
            "time_tolerance": linked.time_tolerance,
        }
        for slot in _LINKED_BREATH_SLOTS:
            breath = linked.breaths.get(slot)
            row[f"{slot}_start_time"] = None if breath is None else breath.start_time
            row[f"{slot}_end_time"] = None if breath is None else breath.end_time
            row[f"{slot}_extremum_time"] = (
                None if breath is None else breath.extremum_time
            )
            row[f"{slot}_extremum_index"] = (
                None if breath is None else breath.extremum_index
            )
        rows.append(row)
    return rows


def parameter_results_to_rows_and_archive(
    parameter_results: Iterable[ParameterResult],
    *,
    archive_filename: str = "parameter_result_arrays.npz",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Split `ParameterResult`s into CSV-ready rows and an array archive.

    Scalar results are unchanged (`value` stays inline). An array-valued
    result gets a deterministic, collision-safe archive key
    (`<sanitized name>_<occurrence index>`); its row swaps the raw `value`
    for `value_file`/`array_key`/`shape`/`dtype` columns instead of
    serializing the array into a CSV cell. A non-scalar `metadata["time"]`
    (e.g. per-breath/per-pixel timing) moves into the same archive under
    `<array_key>_time`, leaving a short reference in the row's metadata.
    """

    rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    occurrence: dict[str, int] = {}

    for parameter in parameter_results:
        row = parameter.to_dict()
        if parameter.is_scalar:
            rows.append(row)
            continue

        array_key = _next_array_key(parameter.name, occurrence)

        value = np.asarray(parameter.value)
        archive[array_key] = value
        row["value"] = None
        row["value_file"] = archive_filename
        row["array_key"] = array_key
        row["shape"] = list(value.shape)
        row["dtype"] = str(value.dtype)

        metadata = dict(row["metadata"])
        time_value = metadata.get("time")
        if time_value is not None and np.ndim(time_value) > 0:
            time_array_key = f"{array_key}_time"
            archive[time_array_key] = np.asarray(time_value, dtype=float)
            metadata["time"] = f"npz:{time_array_key}"
            row["time_array_key"] = time_array_key
        row["metadata"] = metadata

        rows.append(row)

    return rows, archive


def interval_data_to_rows_and_archive(
    interval_data: Iterable[IntervalData],
    *,
    archive_filename: str = "interval_data_arrays.npz",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Turn `IntervalData` results into CSV rows, one row per interval, an
    archive for values that are arrays, and one description per result.

    Each row gives the result (name, modality, unit, ...), the interval
    (position, start and end time, ...) and its value. A number stays in the
    ``value`` column. Values that are arrays (a pixel map per breath, for
    example) are stacked into one array in the archive under ``array_key``;
    the row's ``array_index`` says which slice belongs to its interval. A
    missing value (``None``) next to array values gets a row with no value
    and no slice.

    The third list holds one entry per result with its metadata (the
    settings it was made with, the axes of its arrays, ...), so this is
    written once rather than on every row. Each row's ``result_index`` points
    to its entry.

    Raises:
        ValueError: If the array values of one result do not all have the
            same shape, so they cannot be stacked.
    """

    rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    descriptions: list[dict[str, Any]] = []
    occurrence: dict[str, int] = {}

    for result_index, item in enumerate(interval_data):
        values = item.values
        array_positions: dict[int, int] = {}
        array_key = None
        if values is not None:
            arrays = [
                (position, np.asarray(value))
                for position, value in enumerate(values)
                if np.ndim(value) > 0
            ]
            if arrays:
                shapes = {array.shape for _, array in arrays}
                if len(shapes) != 1:
                    raise ValueError(
                        f"IntervalData {item.name!r} holds arrays of different "
                        f"shapes {sorted(shapes)}; they cannot be stored as one "
                        "array."
                    )
                array_key = _next_array_key(item.name, occurrence)
                archive[array_key] = np.stack([array for _, array in arrays])
                array_positions = {
                    position: slice_index
                    for slice_index, (position, _) in enumerate(arrays)
                }

        descriptions.append(
            {
                "result_index": result_index,
                "name": item.name,
                "modality": item.modality,
                "category": item.category,
                "unit": item.unit,
                "method": item.method,
                "interval_count": len(item),
                "value_file": archive_filename if array_key is not None else None,
                "array_key": array_key,
                "metadata": dict(item.metadata),
            }
        )

        for position, interval in enumerate(item.intervals):
            row: dict[str, Any] = {
                "result_index": result_index,
                "name": item.name,
                "modality": item.modality,
                "category": item.category,
                "unit": item.unit,
                "method": item.method,
                "interval_index": position,
                "interval_name": interval.name,
                "interval_modality": interval.modality,
                "interval_label": interval.label,
                "start_time": interval.start_time,
                "end_time": interval.end_time,
                "value": None,
            }
            if position in array_positions:
                row["value_file"] = archive_filename
                row["array_key"] = array_key
                row["array_index"] = array_positions[position]
            elif values is not None:
                value = values[position]
                row["value"] = value.item() if isinstance(value, np.generic) else value
            rows.append(row)

    return rows, archive, descriptions


def pixel_masks_to_rows_and_archive(
    masks: Iterable[PixelMask],
    *,
    archive_filename: str = "pixel_masks.npz",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Turn `PixelMask` objects into CSV rows, one row per mask, and an
    archive holding each mask's (row, column) grid under ``array_key``. The
    ``metadata`` column keeps the settings the mask was made with."""

    rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    occurrence: dict[str, int] = {}

    for mask in masks:
        array_key = _next_array_key(mask.name, occurrence)
        archive[array_key] = mask.values
        rows.append(
            {
                "name": mask.name,
                "modality": mask.modality,
                "method": mask.method,
                "rows": mask.shape[0],
                "columns": mask.shape[1],
                "included_pixel_count": mask.included_pixel_count,
                "value_file": archive_filename,
                "array_key": array_key,
                "metadata": dict(mask.metadata),
            }
        )

    return rows, archive


def _next_array_key(name: str, occurrence: dict[str, int]) -> str:
    """A new archive key for ``name``: the cleaned-up name plus a count.

    The count is kept per cleaned-up name, not per original name, because
    two names can clean up to the same text (``"pixel-tiv"`` and
    ``"pixel tiv"`` both give ``"pixel_tiv"``) and must still get different
    keys, or one array would overwrite the other.
    """

    base = _sanitize_array_key(name)
    index = occurrence.get(base, 0)
    occurrence[base] = index + 1
    return f"{base}_{index}"


def _sanitize_array_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "result"
