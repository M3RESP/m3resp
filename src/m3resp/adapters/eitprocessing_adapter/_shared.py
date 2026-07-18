"""Standalone helper functions for `EITProcessingAdapter`."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError
from m3resp.data import ParameterResult, Signal
from m3resp.data.signals import Modality, ProcessingState


def _optional_dependency_error() -> OptionalDependencyError:
    return OptionalDependencyError(
        "EIT support requires the optional dependency `eitprocessing`. "
        'Install with `pip install "m3resp[eit]"`.'
    )


def _require_eit_sequence(sequence: Any) -> None:
    if not hasattr(sequence, "eit_data") or not hasattr(sequence, "continuous_data"):
        raise TypeError("EIT preprocessing expects an eitprocessing Sequence.")
    if "raw" not in sequence.eit_data:
        raise KeyError("EIT preprocessing requires sequence.eit_data['raw'].")


def add_to_collection(collection: Any, value: Any) -> None:
    try:
        collection.add(value, overwrite=True)
    except TypeError:
        collection.add(value)


def _breath_intervals_to_dicts(breath_intervals: Any) -> list[dict[str, Any]]:
    return [
        {
            "start_time": breath.start_time,
            "end_time": breath.end_time,
            "peak_time": getattr(breath, "middle_time", None),
            "source": "eitprocessing.BreathDetection",
        }
        for breath in breath_intervals.values
    ]


def continuous_data_to_signal(
    obj: Any,
    *,
    modality: Modality,
    channel: str | None,
    processing_state: ProcessingState,
    source: str | None = None,
    method: str | None = None,
) -> Signal:
    """Convert an `eitprocessing.ContinuousData`-shaped object to a `Signal`."""

    return Signal(
        values=obj.values,
        time=obj.time,
        sample_frequency=getattr(obj, "sample_frequency", None),
        unit=getattr(obj, "unit", None),
        name=getattr(obj, "name", None) or getattr(obj, "label", None),
        modality=modality,
        channel=channel,
        processing_state=processing_state,
        source=source,
        method=method,
    )


def _sparse_data_to_parameters(
    obj: Any, *, modality: str, method: str
) -> list[ParameterResult]:
    """Convert an `eitprocessing.SparseData`-shaped object (one value per
    breath) into one `ParameterResult` per non-NaN sample.

    Per-breath timing is usually a scalar, but pixel-resolved results (e.g.
    pixel TIV) carry a full array (row, column, ...) per breath. Both shapes
    are preserved; the array case is never truncated to a single float.
    """

    values = np.asarray(obj.values)
    times = np.asarray(obj.time, dtype=object)
    name = getattr(obj, "name", None) or getattr(obj, "label", None) or "parameter"
    unit = getattr(obj, "unit", None)

    results: list[ParameterResult] = []
    for index, value in enumerate(values):
        if np.ndim(value) == 0 and np.isnan(value):
            continue
        metadata: dict[str, Any] = {}
        if index < len(times):
            time_entry = np.asarray(times[index])
            if time_entry.ndim == 0:
                metadata["time"] = float(time_entry)
            else:
                metadata["time"] = time_entry.tolist()
                metadata["time_shape"] = list(time_entry.shape)
                metadata["time_axes"] = ["row", "column"][: time_entry.ndim]
        results.append(
            ParameterResult(
                name=name,
                value=value,
                modality=modality,
                unit=unit,
                breath_id=str(index),
                method=method,
                metadata=metadata,
            )
        )
    return results
