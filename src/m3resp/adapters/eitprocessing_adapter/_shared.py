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


def filter_pixels_preserving_gaps(
    pixel_impedance: Any,
    *,
    operation: str,
    apply: Callable[[np.ndarray], Any],
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Filter the measured pixels with `apply`; leave unmeasured pixels missing.

    A pixel that was never measured - outside the electrode plane, or switched
    off - is NaN for the whole recording, and must still be NaN afterwards.
    Replacing it with zero would enter a real impedance reading where there was
    no measurement, and everything downstream would treat it as data.

    A Butterworth filter cannot be evaluated on NaN, so the unmeasured pixels
    are given a placeholder for the duration of the call and set back to NaN in
    the result. Nothing leaks between pixels: `apply` filters along time
    (axis 0) and each pixel's time course is filtered independently, so a
    placeholder can never reach a measured pixel.

    A pixel that was measured but lost *some* samples is a different case: it
    cannot be filtered without inventing the missing stretch, and quietly
    emptying it would lose a real pixel. That is rejected here instead.

    Pass the `captures` dict `apply` writes its diagnostics into to have the
    unmeasured pixels blanked there too, so a diagnostic plot shows them as
    absent rather than as a flat zero trace.
    """

    values = np.asarray(pixel_impedance, dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return np.asarray(apply(values))

    # Axis 0 is time; the remaining axes are the pixel grid.
    n_samples = values.shape[0]
    missing_per_pixel = missing.sum(axis=0)
    partly_missing = (missing_per_pixel > 0) & (missing_per_pixel < n_samples)
    if partly_missing.any():
        examples = ", ".join(
            "(" + ", ".join(str(int(axis)) for axis in position) + ")"
            for position in np.argwhere(partly_missing)[:5]
        )
        raise ValueError(
            f"{operation}: {int(partly_missing.sum())} pixel(s) are missing "
            f"samples for part of the recording only (e.g. at {examples}). "
            "A Butterworth filter cannot span a gap, so these cannot be "
            "filtered without inventing the missing samples. Repair or drop "
            "those pixels first. Pixels missing for the whole recording need "
            "no action - they stay missing through the filter."
        )

    unmeasured = missing_per_pixel == n_samples
    placeholder = values.copy()
    placeholder[:, unmeasured] = 0.0
    filtered = np.array(apply(placeholder), dtype=float, copy=True)
    filtered[:, unmeasured] = np.nan

    # The placeholder also reached any full-size array `apply` recorded as a
    # diagnostic (the unfiltered and filtered pixel data); blank it there too
    # so no capture claims a reading for a pixel that was never measured.
    for key, captured in list((captures or {}).items()):
        recorded = np.asarray(captured)
        if recorded.shape == values.shape:
            blanked = recorded.astype(float, copy=True)
            blanked[:, unmeasured] = np.nan
            captures[key] = blanked  # type: ignore[index]

    return filtered


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
    name: str | None = None,
) -> Signal:
    """Convert an `eitprocessing.ContinuousData`-shaped object to a `Signal`.

    Everything `eitprocessing` emits through this path is an impedance
    (global or pixel-resolved), so ``category`` defaults accordingly; pass it
    explicitly for a channel that measures something else.

    For anything other than ``processing_state="raw"``, ``name`` must be
    passed explicitly - ``obj``'s own ``.name``/``.label`` is not trusted for
    a transformed signal. This is a deliberate guard: upstream filter
    operations (e.g. `eitprocessing`'s `MDNFilter.apply`) deep-copy their raw
    input and only overwrite attributes passed as explicit kwargs, so a
    caller that forgets to pass `name`/`label` through to the *upstream*
    call gets an object whose `.name` still silently says `"raw"` - see the
    `eit.mdn_filter` regression this guard was added for. Raw data is exempt
    because its `.name`/`.label` genuinely comes from the loader, not from a
    copy-then-partially-overwrite operation.
    """

    if processing_state == "raw":
        resolved_name = (
            name or getattr(obj, "name", None) or getattr(obj, "label", None)
        )
    elif name is not None:
        resolved_name = name
    else:
        raise ValueError(
            "continuous_data_to_signal: `name` must be passed explicitly "
            f"for processing_state={processing_state!r} - obj.name/obj.label "
            "cannot be trusted for a transformed signal, since upstream "
            "filter operations may silently leave a stale value there."
        )

    return Signal(
        values=obj.values,
        time=obj.time,
        sample_frequency=getattr(obj, "sample_frequency", None),
        unit=getattr(obj, "unit", None),
        name=resolved_name,
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
