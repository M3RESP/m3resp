"""Shared constants/helpers for the emg processing modules.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/helper_functions/math_operations.py::bell_curve;
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.core.exceptions import (
    MutuallyExclusiveArgsError,
    OptionalDependencyError,
    UnresolvedChannelError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

POSTPROCESSING_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "baseline": ("moving_baseline", "slopesum_baseline"),
    "event_detection": (
        "find_occluded_breaths",
        "onoffpeak_baseline_crossing",
        "onoffpeak_slope_extrapolation",
        "detect_ventilator_breath",
        "detect_emg_breaths",
    ),
    "features": (
        "time_to_peak",
        "pseudo_slope",
        "amplitude",
        "time_product",
        "area_under_baseline",
        "respiratory_rate",
    ),
    "quality_assessment": (
        "snr_pseudo",
        "pocc_quality",
        "interpeak_dist",
        "percentage_under_baseline",
        "detect_local_high_aub",
        "detect_extreme_time_products",
        "detect_non_consecutive_manoeuvres",
        "evaluate_bell_curve_error",
        "evaluate_event_timing",
        "evaluate_respiratory_rates",
    ),
}


_POSTPROCESSING_MODULES = {
    "baseline": "m3resp.emg.processing.baseline",
    "event_detection": "resurfemg.postprocessing.event_detection",
    "features": "resurfemg.postprocessing.features",
    "quality_assessment": "resurfemg.postprocessing.quality_assessment",
}


def _require_emg_recording(recording: Any) -> None:  # noqa: ANN401
    if not isinstance(recording, dict) or "array" not in recording:
        msg = "EMG preprocessing expects a ReSurfEMG recording dict."
        raise TypeError(msg)
    if "metadata" not in recording:
        msg = "EMG preprocessing expects recording metadata."
        raise TypeError(msg)


def _emg_optional_dependency_error() -> OptionalDependencyError:
    return OptionalDependencyError(
        "EMG postprocessing requires the optional dependency `resurfemg`. "
        'Install with `pip install "m3resp[emg]"`.'
    )


def _require_1d_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        msg = f"{name} must be a 1D array; got shape {array.shape}."
        raise ValueError(msg)
    return array


def _require_index_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        msg = f"{name} must be a 1D array of indices; got shape {array.shape}."
        raise ValueError(msg)
    return array.astype(int)


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer; got {value!r}."
        raise ValueError(msg)


def _require_percentile(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 100.0):
        msg = f"{name} must be between 0 and 100; got {value!r}."
        raise ValueError(msg)


def _require_finite_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        msg = f"{name} must be a finite, positive number; got {value!r}."
        raise ValueError(msg)


def _require_equal_length(*named_arrays: tuple[str, np.ndarray]) -> None:
    lengths = {name: len(array) for name, array in named_arrays}
    if len(set(lengths.values())) > 1:
        msg = f"Arrays must have equal length; got {lengths}."
        raise ValueError(msg)


def _mask_invalid(values: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """[NOT APPROVED] Mask invalid values with NaNs.

    Replace entries at invalid breath positions with NaN, preserving
    array length/index alignment with `peak_indices`.

    `valid_peaks` (from `onoff_from_baseline_crossings`) flags breaths
    whose onset/offset window overlaps a neighboring breath or was never found;
    letting those through unmasked would make an overlapping/degenerate
    window masquerade as a real measurement.
    """
    array = np.array(values, dtype=float, copy=True)
    valid = np.asarray(validity, dtype=bool)
    _require_equal_length(("values", array), ("validity", valid))
    array[~valid] = np.nan
    return array


def _require_integer_valued_sample_frequency(sample_frequency: float) -> int:
    """Normalize the sample frequency to integer type.

    Converts the sample frequency to integer type, for use in methods that require
    an exact integer value (e.g., for pandas rolling window calculations).

    Raises a ValueError if the sample frequency is not a finite positive number
    or if it is not an exact integer value.
    """
    if not np.isfinite(sample_frequency) or sample_frequency <= 0:
        msg = f"sample_frequency must be finite and positive; got {sample_frequency!r}."
        raise MutuallyExclusiveArgsError(msg)
    if float(sample_frequency).is_integer():
        return int(sample_frequency)
    msg = (
        "sample_frequency must be an exact integer value for this operation;"
        f" got {sample_frequency!r}."
    )
    raise ValueError(msg)


def _computed_category(postprocessed: dict[str, Any], category: str) -> dict[str, Any]:
    if not isinstance(postprocessed, dict):
        return {}
    return dict(postprocessed.get("computed", {}).get(category, {}))


def _as_parameter_value(value: Any) -> float | np.ndarray:  # noqa: ANN401
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return np.asarray(value)


def _normalize_selected_postprocessing(
    selected_functions: dict[str, dict[str, bool]] | None,
) -> set[tuple[str, str]]:
    if selected_functions is None:
        return {
            (category, function_name)
            for category, functions in POSTPROCESSING_FUNCTIONS.items()
            for function_name in functions
        }

    selected: set[tuple[str, str]] = set()
    for category, functions in POSTPROCESSING_FUNCTIONS.items():
        configured = selected_functions.get(category, {})
        for function_name in functions:
            if configured.get(function_name, False):
                selected.add((category, function_name))
    return selected


def _category_for_function(function_name: str) -> str | None:
    for category, functions in POSTPROCESSING_FUNCTIONS.items():
        if function_name in functions:
            return category
    return None


def _missing_postprocessing_dependency() -> str | None:
    for module_name in _POSTPROCESSING_MODULES.values():
        try:
            import_module(module_name)
        except ImportError:
            return (
                "Optional dependency `resurfemg` is not installed; "
                'install with `pip install "m3resp[emg]"` to compute this function.'
            )
    return None


def _unavailable_postprocessing_result(
    *,
    selected: set[tuple[str, str]],
    peak_indices: np.ndarray,
    computed: dict[str, Any],
    reason: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "available": {},
        "computed": computed,
        "skipped": {
            f"{category}.{function_name}": reason
            for category, function_name in sorted(selected)
        },
        "peak_indices": peak_indices,
        "settings": settings,
    }


#: Channel names that mark a heart (ECG) reference channel rather than EMG.
_ECG_LABELS = ("ecg", "ekg")


def _choose_emg_channel(n_channels: int, labels: Sequence[str] | None) -> int:
    """Pick the EMG channel to analyse when the user did not name one.

    - A recording with one channel: that channel.
    - Otherwise, channels whose name says ECG/EKG are left out. If exactly one
      channel is left, that one is used.
    - Otherwise it is unclear which channel is the breathing muscle, so this
      raises `UnresolvedChannelError` and asks for ``channel=`` instead of
      guessing.
    """
    if n_channels == 1:
        return 0
    names = list(labels or [])
    if len(names) == n_channels:
        candidates = [
            index
            for index, name in enumerate(names)
            if str(name).strip().lower() not in _ECG_LABELS
        ]
        if len(candidates) == 1:
            return candidates[0]
    channel_list = ", ".join(
        f"{index} ({names[index]!r})" if index < len(names) else str(index)
        for index in range(n_channels)
    )

    msg = (
        f"This EMG recording has {n_channels} channels ({channel_list}) and it is "
        "not clear which one is the breathing muscle. Pass the channel number, "
        "for example preprocess_emg(channel=1), or "
        'run_pipeline("emg", config={"preprocess": {"channel": 1}}).'
    )
    raise UnresolvedChannelError(msg)
