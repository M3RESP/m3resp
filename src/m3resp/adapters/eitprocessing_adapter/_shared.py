"""Standalone helper functions for `EITProcessingAdapter`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from inspect import Parameter, signature
from typing import Any

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError
from m3resp.data import IntervalData, PixelMap, Signal
from m3resp.data.events import BreathEvent, coerce_breath_events
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
    """Return one row per EIT breath, preserving times in seconds.

    Read ``breath_intervals.values`` in order and map each breath's
    ``middle_time`` to ``extremum_time``, using None when it is unavailable.
    Rows also carry the source ``"eitprocessing.BreathDetection"``.
    """

    return [
        {
            "start_time": breath.start_time,
            "end_time": breath.end_time,
            "extremum_time": getattr(breath, "middle_time", None),
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


def breath_intervals_to_breath_events(breath_intervals: Any) -> list[BreathEvent]:
    """Convert eitprocessing's detected breaths to m3resp ``BreathEvent`` objects.

    Args:
        breath_intervals: eitprocessing ``IntervalData`` containing breaths,
            each with start, middle and end times in seconds.

    Returns:
        list[BreathEvent]: EIT breaths in input order. The middle time becomes
            ``extremum_time`` and timing retains the input's time axis.
    """

    return coerce_breath_events(
        _breath_intervals_to_dicts(breath_intervals),
        modality="eit",
        source="eitprocessing.BreathDetection",
    )


def sparse_data_to_interval_data(
    obj: Any,
    breaths: list[BreathEvent],
    *,
    modality: str,
    method: str,
    metadata: dict[str, Any] | None = None,
    as_pixel_maps: bool = False,
) -> IntervalData:
    """Pair eitprocessing's per-breath values with their detected breaths.

    Args:
        obj: A result with ``values``, an optional ``unit`` and a ``label`` or
            ``name``. Values have shape ``(breath, ...)`` and are converted
            to floats, including NaNs.
        breaths: Breaths used to compute the values, in the same order.
            Their times supply the result's timing; ``obj.time`` is unused.
        modality: Device or technique the values came from.
        method: Name of the method that computed the values.
        metadata: Additional result information, copied into the result.
        as_pixel_maps: Wrap each row-column grid in ``PixelMap`` when true.
            False retains the numeric array.

    Returns:
        IntervalData: One value per breath, with category ``'impedance'`` and
            the input unit. Breath objects and missing NaN values are kept.

    Raises:
        ValueError: If value and breath counts differ, numeric conversion
            fails, or a pixel value has other than two dimensions.
        TypeError: If values have an unsupported type or an item in ``breaths``
            is other than an ``Interval``.
    """

    values = np.asarray(obj.values, dtype=float)
    if len(values) != len(breaths):
        raise ValueError(
            f"Got {len(values)} per-breath values but {len(breaths)} breaths; "
            "the breaths must be the ones the values were computed over."
        )
    name = getattr(obj, "label", None) or getattr(obj, "name", None) or "parameter"
    unit = getattr(obj, "unit", None)
    per_breath: list[Any] | np.ndarray = values
    if as_pixel_maps:
        per_breath = [
            PixelMap(
                name=name,
                values=grid,
                modality=modality,
                category="impedance",
                unit=unit,
                method=method,
            )
            for grid in values
        ]
    return IntervalData(
        name=name,
        modality=modality,
        intervals=list(breaths),
        values=per_breath,
        category="impedance",
        unit=unit,
        method=method,
        metadata=dict(metadata or {}),
    )
