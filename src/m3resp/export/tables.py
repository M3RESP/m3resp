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
    """Convert linked breaths into one table row per link.

    Args:
        linked_breaths: Linked breath groups in the desired row order.

    Returns:
        list[dict[str, Any]]: Rows with modalities, confidence and time tolerance
            (seconds). Each row includes start/end/extremum times in seconds
            and the extremum sample index for each of ``eit``, ``emg`` and
            ``ventilator``, prefixed by the modality name. Missing breaths or
            turning points give None. Sample indices refer to each breath's
            own signal. Other modality names appear in ``modalities``.
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
    """Prepare parameter table rows and numeric arrays for export.

    Scalar values stay in their rows. Array values are indexed by unique
    keys made from the result name and an occurrence count. Array-valued
    ``metadata['time']`` is also added to the archive and referenced from
    the row.

    Args:
        parameter_results: Results to export, in table order.
        archive_filename: Filename used in rows that reference array values.

    Returns:
        tuple: Rows in input order and a dictionary of arrays for
            ``numpy.savez_compressed``. Array rows contain ``value=None``,
            ``value_file``, ``array_key``, ``shape`` and ``dtype``. Names,
            units and other descriptive fields are retained.
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
    """Prepare one table row per interval, array values and result descriptions.

    Scalar values stay in the ``value`` column. Arrays and pixel maps of the
    same shape are stacked per result; ``array_key`` and ``array_index`` link
    each row to its slice. Missing scalar values remain in their rows.

    Args:
        interval_data: Results with one value per interval, in result order.
            Interval times are in seconds on the input's clock; array axes
            follow each result's metadata.
        archive_filename: Filename used in rows that reference array values.

    Returns:
        tuple: Table rows in result and interval order, a dictionary of
            stacked arrays for ``numpy.savez_compressed``, and one metadata
            description per result. Each row's ``result_index`` selects its
            description. Units, interval names and start-end times are kept.
            A ``None`` value has ``value=None`` and no array slice reference.

    Raises:
        ValueError: If array values within a result have different shapes.
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
    """Prepare pixel-mask table rows and grids for export.

    Args:
        masks: Masks to export, in table order. Grids have row-column axes,
            positive weights for included pixels and NaN for excluded pixels.
        archive_filename: Filename used in the rows to reference mask grids.

    Returns:
        tuple: One row per mask and a dictionary of grids for
            ``numpy.savez_compressed``. Each row includes ``array_key``,
            ``value_file``, dimensions, included-pixel count and metadata.
            Grids retain their weights and NaNs.
    """

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
    """Return a cleaned name with an occurrence count and increment that count.

    Counts are shared by names that clean to the same text, so each key in
    an archive is unique.
    """

    base = _sanitize_array_key(name)
    index = occurrence.get(base, 0)
    occurrence[base] = index + 1
    return f"{base}_{index}"


def _sanitize_array_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "result"
