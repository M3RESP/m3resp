"""Standalone helper functions for `EITProcessingAdapter`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from inspect import Parameter, signature
from typing import Any

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError
from m3resp.data import IntervalData, PixelMap
from m3resp.data.events import BreathEvent, coerce_breath_events
from m3resp.eit.processing import BreathDetectionSettings


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


#: Highest order of a Butterworth filter of EIT images, as in eitprocessing.
MAX_EIT_BUTTERWORTH_ORDER = 10


def check_eit_butterworth_order(order: int) -> None:
    """Refuse a Butterworth order outside 1 to 10 for EIT images.

    The filter runs forwards and backwards, so the effective order is twice
    the given order. Higher orders can give unstable or wrong results.

    Args:
        order (int): The order of the filter.

    Raises:
        ValueError: If ``order`` is below 1 or above 10.
    """

    if not 1 <= order <= MAX_EIT_BUTTERWORTH_ORDER:
        raise ValueError(
            "The order of a Butterworth filter of EIT images must be from 1 to "
            f"{MAX_EIT_BUTTERWORTH_ORDER}, got {order}. Higher orders can give "
            "unstable results."
        )


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
    """Return one row per eitprocessing breath, preserving times in seconds.

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


def breath_intervals_to_breath_events(breath_intervals: Any) -> list[BreathEvent]:
    """Return detected EIT breaths as m3resp ``BreathEvent`` objects.

    Args:
        breath_intervals: The breaths, in one of three forms: an m3resp
            ``IntervalData`` of ``BreathEvent``s (``eit.detect_breaths``), a
            list of ``BreathEvent``s, or an eitprocessing ``IntervalData`` of
            ``Breath``s. Times are in seconds.

    Returns:
        list[BreathEvent]: EIT breaths in input order, on the time axis of the
            input. m3resp breaths are returned as they are; for eitprocessing
            breaths the middle time becomes ``extremum_time``.
    """

    if isinstance(breath_intervals, IntervalData):
        return coerce_breath_events(breath_intervals.intervals, modality="eit")
    if isinstance(breath_intervals, list):
        return coerce_breath_events(breath_intervals, modality="eit")
    return coerce_breath_events(
        _breath_intervals_to_dicts(breath_intervals),
        modality="eit",
        source="eitprocessing.BreathDetection",
    )


def breaths_to_interval_data(
    breaths: list[BreathEvent], *, name: str = "eit_breaths"
) -> IntervalData:
    """Return EIT breaths as an m3resp ``IntervalData`` without values.

    Args:
        breaths (list[BreathEvent]): The breaths, in time order.
        name (str): The name of the result.

    Returns:
        IntervalData: The breaths in ``intervals``, modality ``"eit"``,
            category ``"breath"`` and method
            ``"m3resp.eit.processing.find_breaths"``.
    """

    return IntervalData(
        name=name,
        modality="eit",
        intervals=list(breaths),
        category="breath",
        method="m3resp.eit.processing.find_breaths",
    )


def breaths_to_eitprocessing(
    breaths: list[BreathEvent],
    settings: BreathDetectionSettings,
    *,
    label: str,
    derived_from: list[Any],
) -> Any:
    """Return EIT breaths as an eitprocessing ``IntervalData`` of ``Breath``s.

    This is the form ``BreathDetection.find_breaths`` gives, for a
    ``Sequence``.

    Args:
        breaths (list[BreathEvent]): The breaths, in time order.
        settings (BreathDetectionSettings): The settings the breaths were
            found with; stored in ``parameters``.
        label (str): The label of the result.
        derived_from (list[Any]): The data the breaths were found in.

    Returns:
        Any: An eitprocessing ``IntervalData`` with category ``"breath"``.
            ``extremum_time`` becomes the ``middle_time`` of each ``Breath``.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
    """

    Breath, UpstreamIntervalData = _lazy_import(
        "eitprocessing.datahandling.breath.Breath",
        "eitprocessing.datahandling.intervaldata.IntervalData",
    )
    return UpstreamIntervalData(
        label=label,
        name="Breaths as determined by m3resp.eit.processing.find_breaths",
        unit=None,
        category="breath",
        intervals=[(breath.start_time, breath.end_time) for breath in breaths],
        values=[
            Breath(breath.start_time, breath.extremum_time, breath.end_time)
            for breath in breaths
        ],
        parameters={BreathDetectionSettings: asdict(settings)},
        derived_from=list(derived_from),
    )


def as_breath_detection(detector: Any) -> Any:
    """Return a breath detector as an eitprocessing ``BreathDetection``.

    The eitprocessing TIV, EELI and pixel-TIV calculations take an
    eitprocessing ``BreathDetection``. It is made from the same settings, so
    it finds the same breaths as ``m3resp.eit.processing.find_breaths``.

    Args:
        detector: ``BreathDetectionSettings``, or an eitprocessing
            ``BreathDetection``.

    Returns:
        Any: An eitprocessing ``BreathDetection`` with the same settings; an
            eitprocessing detector is returned as it is.

    Raises:
        OptionalDependencyError: If eitprocessing is not installed.
    """

    if not isinstance(detector, BreathDetectionSettings):
        return detector
    (BreathDetection,) = _lazy_import(
        "eitprocessing.features.breath_detection.BreathDetection"
    )
    return BreathDetection(**asdict(detector))


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
