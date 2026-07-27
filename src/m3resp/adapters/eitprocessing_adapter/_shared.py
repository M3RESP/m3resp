"""Standalone helper functions for `EITProcessingAdapter`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from inspect import Parameter, signature
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


def _import_attr(dotted_path: str) -> Any:
    module_path, _, attr_name = dotted_path.rpartition(".")
    # Use the `__import__` builtin (as `from module import name` does), not
    # `importlib.import_module` - the latter bypasses `builtins.__import__`
    # via internal bootstrap machinery, which breaks tests that monkeypatch
    # `builtins.__import__` to simulate `eitprocessing` being uninstalled.
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name)


def _lazy_import(
    *dotted_paths: str,
    error: Callable[[], OptionalDependencyError] | None = None,
) -> tuple[Any, ...]:
    """Import one or more `eitprocessing` attributes on demand.

    Centralizes the try/import/except-ImportError dance every adapter method
    needs to keep `eitprocessing` optional, so each method states only which
    attributes it needs. Raises a consistent `OptionalDependencyError`
    (or `error()` if given, for callers with a more specific message).
    """

    try:
        return tuple(_import_attr(path) for path in dotted_paths)
    except ImportError as exc:
        raise (error() if error is not None else _optional_dependency_error()) from exc


def _require_eit_sequence(sequence: Any) -> None:
    if not hasattr(sequence, "eit_data") or not hasattr(sequence, "continuous_data"):
        raise TypeError("EIT preprocessing expects an eitprocessing Sequence.")
    if "raw" not in sequence.eit_data:
        raise KeyError("EIT preprocessing requires sequence.eit_data['raw'].")


def add_to_collection(collection: Any, value: Any) -> None:
    """Add a value while supporting old collections without ``overwrite``."""

    parameters: Iterable[Parameter]
    try:
        parameters = signature(collection.add).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_overwrite = any(
        parameter.name == "overwrite" or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_overwrite:
        collection.add(value, overwrite=True)
    else:
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
    category: str | None = "impedance",
) -> Signal:
    """Convert an `eitprocessing.ContinuousData`-shaped object to a `Signal`.

    Everything `eitprocessing` emits through this path is an impedance
    (global or pixel-resolved), so ``category`` defaults accordingly; pass it
    explicitly for a channel that measures something else.
    """

    return Signal(
        values=obj.values,
        time=obj.time,
        sample_frequency=getattr(obj, "sample_frequency", None),
        unit=getattr(obj, "unit", None),
        name=getattr(obj, "name", None) or getattr(obj, "label", None),
        modality=modality,
        category=category,
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
