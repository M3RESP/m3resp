"""Shared constants/helpers for the ReSurfEMGAdapter mixin modules."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError

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

    import pandas as pd

    with open(path, encoding="utf-8", errors="replace") as handle:
        header: list[str] = [handle.readline().rstrip("\n") for _ in range(3)]

    msec_per_sample = float(header[1].split()[0])
    fs = 1000.0 / msec_per_sample
    n_channels = int(header[2].split()[0])

    labels: list[str] = []
    units: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
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


def _require_emg_recording(recording: Any) -> None:
    if not isinstance(recording, dict) or "array" not in recording:
        raise TypeError("EMG preprocessing expects a ReSurfEMG recording dict.")
    if "metadata" not in recording:
        raise TypeError("EMG preprocessing expects recording metadata.")


def _emg_optional_dependency_error() -> OptionalDependencyError:
    return OptionalDependencyError(
        "EMG postprocessing requires the optional dependency `resurfemg`. "
        'Install with `pip install "m3resp[emg]"`.'
    )


def _require_1d_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {array.shape}.")
    return array


def _require_index_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array of indices; got shape {array.shape}."
        )
    return array.astype(int)


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def _require_percentile(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 100.0):
        raise ValueError(f"{name} must be between 0 and 100; got {value!r}.")


def _require_finite_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number; got {value!r}.")


def _require_equal_length(*named_arrays: tuple[str, np.ndarray]) -> None:
    lengths = {name: len(array) for name, array in named_arrays}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays must have equal length; got {lengths}.")


def _mask_invalid(values: Any, validity: Any) -> np.ndarray:
    """Replace entries at invalid breath positions with NaN, preserving
    array length/index alignment with `peak_indices`. `valid_peaks` (from
    `onoff_from_baseline_crossings`) flags breaths whose onset/offset window
    overlaps a neighboring breath or was never found; letting those through
    unmasked would make an overlapping/degenerate window masquerade as a
    real measurement."""

    array = np.array(values, dtype=float, copy=True)
    valid = np.asarray(validity, dtype=bool)
    _require_equal_length(("values", array), ("validity", valid))
    array[~valid] = np.nan
    return array


def _require_integer_valued_sample_frequency(sample_frequency: float) -> int:
    """Normalize `sample_frequency` to `int` for upstream calls that need an
    exact integer (e.g. `fs // 200` fed straight into a pandas rolling
    window - see plan/stage2/2_resurfemg_gap_migration_implementation_plan.md
    Phase 0.2). Only an exactly integer-valued float (`2048.0`) is
    normalized; a genuinely fractional frequency is a clear error rather
    than a silent rounding.
    """

    if not np.isfinite(sample_frequency) or sample_frequency <= 0:
        raise ValueError(
            f"sample_frequency must be finite and positive; got {sample_frequency!r}."
        )
    if float(sample_frequency).is_integer():
        return int(sample_frequency)
    raise ValueError(
        "sample_frequency must be an exact integer value for this operation "
        f"(ReSurfEMG requires an int internally); got {sample_frequency!r}."
    )


def _computed_category(postprocessed: dict[str, Any], category: str) -> dict[str, Any]:
    if not isinstance(postprocessed, dict):
        return {}
    return dict(postprocessed.get("computed", {}).get(category, {}))


def _as_parameter_value(value: Any) -> float | np.ndarray:
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
    peak_indices: Any,
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
