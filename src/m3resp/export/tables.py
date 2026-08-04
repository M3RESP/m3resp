"""Convert M3Resp objects into table rows."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.core.events import BreathEvent, Event, event_to_dict

if TYPE_CHECKING:
    from m3resp.data.linked_breath import LinkedBreath
    from m3resp.data.parameters import ParameterResult

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
    ``emg_peak_time``, ...) and left ``None`` when that modality has no
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
            row[f"{slot}_peak_time"] = None if breath is None else breath.peak_time
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

        index = occurrence.get(parameter.name, 0)
        occurrence[parameter.name] = index + 1
        array_key = f"{_sanitize_array_key(parameter.name)}_{index}"

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


def _sanitize_array_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "result"
