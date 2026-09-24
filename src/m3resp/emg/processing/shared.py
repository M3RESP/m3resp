"""Shared constants/helpers for the emg processing modules.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/helper_functions/math_operations.py::derivative, bell_curve;
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `derivative` renamed to `_compute_derivative`
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
import pandas as pd

from m3resp.core.exceptions import MutuallyExclusiveArgsError, OptionalDependencyError

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
    "baseline": "resurfemg.postprocessing.baseline",
    "event_detection": "resurfemg.postprocessing.event_detection",
    "features": "resurfemg.postprocessing.features",
    "quality_assessment": "resurfemg.postprocessing.quality_assessment",
}


def _load_biopac_txt(path: str) -> dict[str, Any]:
    """Load a Biopac/AcqKnowledge tab-delimited ``.txt`` export.

    The header looks like::
        Paw_EMG.gtl
        0.5 msec/sample
        3 channels
        Paw - TSD104A - Blood Pressure, DA100C
        cmH2O
        EMGdi - EMG100C
        mV
        EMGps - EMG100C
        mV
        CH1<TAB>CH2<TAB>CH3
        3571881<TAB>3571887<TAB>3571887
        <numeric data rows...>

    i.e. a title line, a ``msec/sample`` sampling-interval line, a
    ``N channels`` line, then two lines per channel (label, unit), a ``CHn``
    column header, a per-channel sample-count row, and finally the samples.
    Returns the same ``(array, dataframe, metadata)``-shaped dict as
    :meth:`ReSurfEMGAdapter.load`, with ``array`` channel-major
    ``(n_channels, n_samples)`` and ``metadata["fs"]`` populated.
    """
    import pandas as pd  # noqa: PLC0415

    with Path.open(Path(path), encoding="utf-8", errors="replace") as handle:
        header: list[str] = [handle.readline().rstrip("\n") for _ in range(3)]

    msec_per_sample = float(header[1].split()[0])
    fs = 1000.0 / msec_per_sample
    n_channels = int(header[2].split()[0])

    labels: list[str] = []
    units: list[str] = []
    with Path.open(Path(path), encoding="utf-8", errors="replace") as handle:
        for _ in range(3):
            handle.readline()
        for _ in range(n_channels):
            label_line = handle.readline().rstrip("\n")
            unit_line = handle.readline().rstrip("\n")
            # "Paw - TSD104A - Blood Pressure, DA100C" -> "Paw"
            labels.append(label_line.split(" - ")[0].strip())
            units.append(unit_line.strip())

    # 3 title/rate/channel lines + 2 lines per channel + column-header row
    # + per-channel sample-count row precede the numeric samples.
    skiprows = 3 + 2 * n_channels + 2
    dataframe = pd.read_csv(
        path,
        sep="\t",
        skiprows=skiprows,
        names=labels,
        usecols=range(n_channels),
        engine="c",
    )
    array = dataframe.to_numpy(dtype=float).T  # channel-major (n_channels, n_samples)
    metadata = {
        "fs": fs,
        "labels": labels,
        "units": units,
        "file_name": Path(path).name,
        "file_dir": str(Path(path).parent),
        "file_extension": "txt",
    }
    return {"array": array, "dataframe": dataframe, "metadata": metadata}


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


def _validate_incompatible_kwargs(
    arg_name1: str = "Arg1",
    arg_name2: str = "Arg2",
    arg_value1: Any | None = None,
    arg_value2: Any | None = None,
    *,
    default_value: Any | None = None,
    **kwargs,
) -> Any:
    """Validate that two mutually exclusive parameters are not both set.

    Checks that only one of two mutually exclusive parameters is set,
    and returns the value of the one that is set, or a default value if neither is set.
    This is useful for functions that accept parameters under multiple names for
    backward compatibility or convenience: in such cases, the "alternative"
    parameter name is often passed in via ``kwargs``.

    Arguments:
        arg_name1 (str): Name of the first argument.
        arg_name2 (str): Name of the second argument.
        arg_value1 (Any | None): Value of the first argument.
            If not provided, it will be looked up in ``kwargs``.
        arg_value2 (Any | None, optional): Value of the second argument.
            If not provided, it will be looked up in ``kwargs``.
        default_value (Any | None): Default value to use if neither argument is provided
        **kwargs: caller function kwargs.

    Returns:
        Any: The value of the argument that is set, or the default value if neither is set.

    Raises:
        MutuallyExclusiveArgsError: If both arguments are set.

    Example:
        case 1: arg1 is a named argument, arg2 is passed in via kwargs
        ```python
        def my_function(arg1=None, **kwargs):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", arg_value1=arg1, default_value=0.1, **kwargs
            )
            ```

        case 2: both arg1 and arg2 are named arguments
        ```python
        def my_function(arg1=None, arg2=None):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", arg_value1=arg1, arg_value2=arg2, default_value=0.1
            )
        ```

        case 3: both arg1 and arg2 are passed in via kwargs
        ```python
        def my_function(**kwargs):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", default_value=0.1, **kwargs
            )
        ```
    """
    arg_value1 = arg_value1 or kwargs.get(arg_name1)
    arg_value2 = arg_value2 or kwargs.get(arg_name2)
    if arg_value1 is not None and arg_value2 is not None:
        msg_0 = f"{arg_name1} and {arg_name2} cannot both be set at the same time."
        raise MutuallyExclusiveArgsError(msg_0)
    if arg_value1 is not None:
        return arg_value1
    if arg_value2 is not None:
        return arg_value2
    return default_value
    # return arg_value1 or arg_value2 or default_value


def _require_sampling_frequency(sample_frequency: float | None) -> float:
    if sample_frequency is None:
        msg = "sample_frequency is required."
        raise ValueError(msg)
    if not np.isfinite(sample_frequency) or sample_frequency <= 0:
        msg = f"sample_frequency must be finite and positive; got {sample_frequency!r}."
        raise ValueError(msg)
    return float(sample_frequency)


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


def _compute_derivative(
    array: np.ndarray, fs: float, window_length_samples: int | None = None
) -> np.ndarray:
    """Calculate the first derivative of a signal.

    If window_length_samples is given, the signal is smoothed before derivative
    calculation.

    Args:
        array (numpy.ndarray): Signal to calculate the derivative over.
        fs (int): Sampling rate.
        window_length_samples (int, optional): Centralised averaging window
            length in samples.

    Returns:
        numpy.ndarray: The 1st derivative of the signal, length len(signal)-1.
    """
    if window_length_samples is not None:
        array_moving_average = (
            pd.Series(array)
            .rolling(window=window_length_samples, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        derivative = np.diff(array_moving_average) * fs
    else:
        derivative = np.diff(array) * fs
    return derivative


def bell_curve(
    array: np.ndarray, amplitudes: float, time_shift: float, steepness: float
) -> np.ndarray:
    """Calculate a shifted, smoothed and amplified bell curve.

    This function calculates a bell curve on the samples of the input array, shifted by
    the time_shift, amplified by the amplitudes for a standard amplitude of 1.

    Args:
        array (numpy.ndarray): values to calculate the bell_curve for.
        amplitudes (float): Amplitude of the bell-curve.
        time_shift (float): Time shift of the bell-curve along the x-axis.
        steepness (float): Steepness factor of bell-curve.

    Returns:
        numpy.ndarray: Bell curve values.
    """
    return amplitudes * np.exp(-((array - time_shift) ** 2) / steepness**2)
