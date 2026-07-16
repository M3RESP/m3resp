"""Adapter boundary for the upstream `resurfemg` package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.events import coerce_breath_events
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.signals import ProcessingState
from m3resp.processing.filters import harmonic_notch_filter
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
    onoff_from_slope,
)
from m3resp.processing.metrics import (
    amplitude_at_peaks,
    area_under_baseline,
    pseudo_slope,
    respiratory_rate_from_indices,
    time_to_peak,
    window_integral,
)
from m3resp.processing.peaks import (
    detect_emg_breath_peaks,
    detect_occluded_breath_peaks,
    detect_ventilator_breath_peaks,
)
from m3resp.processing.quality import quality_flag_from_result, skipped_quality_flag
from m3resp.processing.windows import rolling_arv

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


class ReSurfEMGAdapter:
    """Thin wrapper around `resurfemg`."""

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, path: str, **kwargs: Any) -> Any:
        """Load EMG data through `resurfemg` or an injected loader."""

        if self._loader is not None:
            return self._loader(path, **kwargs)

        # Biopac/AcqKnowledge tab-delimited text exports (used by the
        # eit_emg_annemijn dataset) are not one of resurfemg's supported
        # extensions and, unlike .npy/.csv, carry their sample rate and channel
        # labels in a text header. Parse them ourselves so `metadata["fs"]`
        # (required by `_preprocess_default`) is populated.
        if str(path).lower().endswith(".txt"):
            return _load_biopac_txt(path)

        try:
            from resurfemg.data_connector.converter_functions import load_file
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG support requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"` or inject a loader.'
            ) from exc

        array, dataframe, metadata = load_file(path, **kwargs)
        return {
            "array": array,
            "dataframe": dataframe,
            "metadata": metadata,
        }

    def preprocess(self, signal: Any, **kwargs: Any) -> Any:
        """Preprocess EMG data through ReSurfEMG or a provided callable."""

        preprocess = kwargs.pop("preprocess", None)
        if preprocess is not None:
            return preprocess(signal, **kwargs)

        return self._preprocess_default(signal, **kwargs)

    def detect_breaths(self, signal: Any, **kwargs: Any) -> list[BreathEvent]:
        """Detect EMG breaths and normalize them into `BreathEvent` objects."""

        detector = kwargs.pop("detector", None)
        if detector is not None:
            detections = detector(signal, **kwargs)
            return coerce_breath_events(
                detections,
                modality="emg",
                source="resurfemg",
            )

        detections = self._detect_breaths_default(signal, **kwargs)
        return coerce_breath_events(
            detections,
            modality="emg",
            source="resurfemg.detect_emg_breaths",
        )

    def compute_features(
        self, signal: Any, events: Sequence[BreathEvent], **kwargs: Any
    ) -> Any:
        """Compute EMG features when an upstream callable is provided."""

        compute = kwargs.pop("compute", None)
        if compute is None:
            raise UnsupportedWorkflowError(
                "EMG feature extraction needs an upstream callable in Stage 1. "
                "Pass `compute=callable`."
            )
        return compute(signal, events, **kwargs)

    # -- Milestone 2.3: conversion to m3resp-native Layer 1 objects -----------
    #
    # These convert `preprocess_emg()`/`postprocess()`'s raw dict outputs into
    # `m3resp.data` objects, so downstream code (and the data model recorder)
    # stops depending on ReSurfEMG's dict layout.

    def to_signals(self, processed_emg: dict[str, Any]) -> list[Signal]:
        """Convert preprocessed EMG channel arrays into `Signal` objects."""

        if not isinstance(processed_emg, dict) or "fs" not in processed_emg:
            raise UnsupportedWorkflowError(
                "to_signals expects processed EMG data from preprocess_emg()."
            )

        fs = float(processed_emg["fs"])
        channel = processed_emg.get("channel")
        channel_name = str(channel) if channel is not None else None

        signals: list[Signal] = []
        channel_sources: list[tuple[str, str, ProcessingState]] = [
            ("raw_channel", "raw_channel", "raw"),
            ("filtered", "filtered", "filtered"),
            ("envelope", "envelope", "processed"),
        ]
        for key, name, processing_state in channel_sources:
            array = processed_emg.get(key)
            if array is None:
                continue
            array = np.asarray(array, dtype=float)
            time = np.arange(array.shape[0], dtype=float) / fs
            signals.append(
                Signal(
                    values=array,
                    time=time,
                    sample_frequency=fs,
                    name=name,
                    modality="emg",
                    channel=channel_name,
                    processing_state=processing_state,
                )
            )
        return signals

    def to_parameters(self, postprocessed: dict[str, Any]) -> list[ParameterResult]:
        """Convert computed EMG features into `ParameterResult` objects."""

        features = _computed_category(postprocessed, "features")
        return [
            ParameterResult(
                name=name,
                value=_as_parameter_value(value),
                modality="emg",
                method=f"resurfemg.{name}",
                metadata={
                    "source_method": f"resurfemg.{name}",
                    "implementation": "m3resp.processing.metrics",
                },
            )
            for name, value in features.items()
        ]

    def to_quality_flags(self, postprocessed: dict[str, Any]) -> list[QualityFlag]:
        """Convert computed EMG quality-assessment results into `QualityFlag`
        objects.

        ReSurfEMG's ``quality_assessment`` functions return heterogeneous
        shapes (booleans, SNR floats, per-breath arrays), so this performs a
        best-effort structural conversion rather than clinical judgment:
        boolean-like results become pass/fail, everything else is recorded as
        an informational flag with its scalar value attached where possible.
        Functions skipped for missing inputs (`postprocessed["skipped"]`)
        become failed, ``warning``-severity flags.
        """

        if not isinstance(postprocessed, dict):
            return []

        quality_results = _computed_category(postprocessed, "quality_assessment")
        flags = [
            quality_flag_from_result(
                name=name,
                result=value,
                modality="emg",
                source_method=f"resurfemg.{name}",
            )
            for name, value in quality_results.items()
        ]
        for name, reason in postprocessed.get("skipped", {}).items():
            flags.append(
                skipped_quality_flag(
                    name=name,
                    modality="emg",
                    reason=str(reason),
                )
            )
        return flags

    def available_postprocessing(self) -> dict[str, list[str]]:
        """Return ReSurfEMG postprocessing functions exposed by M3Resp."""

        return {
            category: list(functions)
            for category, functions in POSTPROCESSING_FUNCTIONS.items()
        }

    def postprocess(
        self,
        processed_emg: Any,
        events: Sequence[BreathEvent] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run default EMG postprocessing and report unavailable input needs."""

        custom = kwargs.pop("postprocess", None)
        if custom is not None:
            return custom(processed_emg, events=events, **kwargs)

        return self._postprocess_default(processed_emg, events=events, **kwargs)

    def run_postprocessing_function(
        self, category: str, function_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Call any exposed `resurfemg.postprocessing` function by name."""

        if function_name not in POSTPROCESSING_FUNCTIONS.get(category, ()):
            raise ValueError(
                f"Unknown ReSurfEMG postprocessing function {category}.{function_name}."
            )

        try:
            module = import_module(_POSTPROCESSING_MODULES[category])
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG postprocessing requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"`.'
            ) from exc

        return getattr(module, function_name)(*args, **kwargs)

    # Named methods for the migrated ECG/baseline/quality operations, with
    # descriptive m3resp argument names converted to the upstream ones and
    # validated before the call. Sample-based arguments (not seconds) are
    # used here for exact upstream behavior; workflow steps convert from
    # seconds and record both the requested seconds and effective sample
    # counts. `_preprocess_default`/`_postprocess_default` and the granular
    # workflow steps call these methods rather than the upstream functions
    # directly, so the monolithic and granular paths share one
    # implementation of each operation.

    def detect_ecg_peaks(
        self,
        signal: Any,
        *,
        sample_frequency: float,
        peak_fraction: float = 0.4,
        peak_width_samples: int | None = None,
        peak_distance_samples: int | None = None,
        bandpass_filter: bool = True,
    ) -> np.ndarray:
        """Detect ECG peak sample indices in `signal` (ECG or an
        ECG-contaminated EMG channel)."""

        try:
            from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        return np.asarray(
            detect_ecg_peaks(
                array,
                fs,
                peak_fraction=peak_fraction,
                peak_width_s=peak_width_samples,
                peak_distance=peak_distance_samples,
                bp_filter=bandpass_filter,
            ),
            dtype=int,
        )

    def gate_ecg(
        self,
        signal: Any,
        peak_indices: Any,
        *,
        gate_width_samples: int = 205,
        fill_method: int = 1,
    ) -> np.ndarray:
        """Gate (remove) ECG peaks from `signal` around `peak_indices`.

        `fill_method`: 0 zeros, 1 interpolation (default upstream), 2 mean
        of a neighboring segment, 3 running-RMS-based replacement.
        """

        try:
            from resurfemg.preprocessing.ecg_removal import gating
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        if fill_method not in (0, 1, 2, 3):
            raise ValueError(
                f"fill_method must be one of 0, 1, 2, 3; got {fill_method!r}."
            )
        if gate_width_samples <= 0:
            raise ValueError(
                f"gate_width_samples must be positive; got {gate_width_samples!r}."
            )
        return np.asarray(
            gating(
                array,
                peaks,
                gate_width=gate_width_samples,
                method=fill_method,
            )
        )

    def wavelet_denoise_ecg(
        self,
        signal: Any,
        peak_indices: Any,
        *,
        sample_frequency: float,
        hard_thresholding: bool = True,
        levels: int = 4,
        wavelet_type: str = "db2",
        fixed_threshold: float = 4.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove ECG artifacts from `signal` via a-trous wavelet shrinkage.

        Returns `(cleaned_signal, decomposition, thresholds, gate_mask)`.
        """

        try:
            from resurfemg.preprocessing.ecg_removal import wavelet_denoising
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        if levels <= 0:
            raise ValueError(f"levels must be positive; got {levels!r}.")
        cleaned, decomposition, thresholds, gate_mask = wavelet_denoising(
            array,
            peaks,
            fs,
            hard_thresholding=hard_thresholding,
            n=levels,
            wavelet_type=wavelet_type,
            fixed_threshold=fixed_threshold,
        )
        return (
            np.asarray(cleaned),
            np.asarray(decomposition),
            np.asarray(thresholds),
            np.asarray(gate_mask),
        )

    def moving_baseline(
        self,
        envelope: Any,
        *,
        window_samples: int,
        step_samples: int,
        percentile: float = 33.0,
    ) -> np.ndarray:
        """Compute a moving baseline over `envelope` (Grasshoff et al. 2021)."""

        try:
            from resurfemg.postprocessing.baseline import moving_baseline
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)
        return np.asarray(
            moving_baseline(
                array,
                window_samples,
                step_samples,
                set_percentile=percentile,
            )
        )

    def slopesum_baseline(
        self,
        envelope: Any,
        *,
        window_samples: int,
        step_samples: int,
        sample_frequency: float,
        percentile: float = 33.0,
        augmented_percentile: float = 25.0,
        moving_average_samples: int | None = None,
        percentile_window_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Compute the slope-sum (augmented) baseline over `envelope`.

        Returns `(baseline, running_mean, running_std, running_series)` -
        `running_series` is upstream's `pandas.Series`, kept for the
        existing compatibility output; use the other three arrays for
        native/export use.
        """

        try:
            from resurfemg.postprocessing.baseline import slopesum_baseline
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)
        _require_percentile("augmented_percentile", augmented_percentile)
        # `slopesum_baseline` derives `ma_window = fs // 2` internally and
        # feeds it straight into a pandas rolling window when
        # `moving_average_samples` is omitted - same int-only constraint as
        # `_require_integer_valued_sample_frequency`'s docstring describes.
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        baseline, running_mean, running_std, running_series = slopesum_baseline(
            array,
            window_samples,
            step_samples,
            fs,
            set_percentile=percentile,
            augm_percentile=augmented_percentile,
            ma_window=moving_average_samples,
            perc_window=percentile_window_samples,
        )
        return (
            np.asarray(baseline),
            np.asarray(running_mean),
            np.asarray(running_std),
            running_series,
        )

    def snr_pseudo(
        self,
        envelope: Any,
        peak_indices: Any,
        baseline: Any,
        *,
        sample_frequency: float,
    ) -> np.ndarray:
        """Pseudo signal-to-noise ratio per peak: peak height relative to a
        local baseline window."""

        try:
            from resurfemg.postprocessing.quality_assessment import snr_pseudo
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        peaks = _require_index_array("peak_indices", peak_indices)
        baseline_array = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        return np.asarray(snr_pseudo(array, peaks, baseline_array, fs))

    def pocc_quality(
        self,
        pressure_signal: Any,
        pocc_peak_indices: Any,
        pocc_end_indices: Any,
        pocc_time_products: Any,
        *,
        dp_up_10_threshold: float = 0.0,
        dp_up_90_threshold: float = 2.0,
        dp_up_90_norm_threshold: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate occlusion-manoeuvre (Pocc) quality (Warnaar et al. 2024).

        Returns `(valid_poccs, criteria_matrix)`; `criteria_matrix` rows are
        `dp_up_10`, `dp_up_90`, `dp_up_90_norm`, in that order.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import pocc_quality
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        pressure = _require_1d_array("pressure_signal", pressure_signal)
        peaks = _require_index_array("pocc_peak_indices", pocc_peak_indices)
        ends = _require_index_array("pocc_end_indices", pocc_end_indices)
        time_products = _require_1d_array("pocc_time_products", pocc_time_products)
        _require_equal_length(
            ("pocc_peak_indices", peaks),
            ("pocc_end_indices", ends),
            ("pocc_time_products", time_products),
        )
        valid_poccs, criteria_matrix = pocc_quality(
            pressure,
            peaks,
            ends,
            time_products,
            dp_up_10_threshold=dp_up_10_threshold,
            dp_up_90_threshold=dp_up_90_threshold,
            dp_up_90_norm_threshold=dp_up_90_norm_threshold,
        )
        return np.asarray(valid_poccs), np.asarray(criteria_matrix)

    def interpeak_distance(
        self,
        ecg_peak_indices: Any,
        emg_peak_indices: Any,
        *,
        threshold: float = 1.1,
    ) -> bool:
        """Whether the median ECG-to-median-EMG interpeak distance ratio is
        within `threshold` (Warnaar et al. 2024)."""

        try:
            from resurfemg.postprocessing.quality_assessment import interpeak_dist
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        ecg_peaks = _require_index_array("ecg_peak_indices", ecg_peak_indices)
        emg_peaks = _require_index_array("emg_peak_indices", emg_peak_indices)
        if len(ecg_peaks) < 2 or len(emg_peaks) < 2:
            raise ValueError(
                "interpeak_distance needs at least two peaks in each of "
                "ecg_peak_indices and emg_peak_indices."
            )
        return bool(interpeak_dist(ecg_peaks, emg_peaks, threshold=threshold))

    def percentage_under_baseline(
        self,
        signal: Any,
        peak_indices: Any,
        start_indices: Any,
        end_indices: Any,
        baseline: Any,
        *,
        sample_frequency: float,
        aub_window_samples: int | None = None,
        reference_signal: Any | None = None,
        aub_threshold: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Percentage area-under-baseline per breath (Warnaar et al. 2024).

        Returns `(valid_time_products, percentages_aub, reference_values)` -
        `reference_values` (upstream's undocumented third return value) are
        the nadir reference levels used per breath.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                percentage_under_baseline,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        starts = _require_index_array("start_indices", start_indices)
        ends = _require_index_array("end_indices", end_indices)
        baseline_array = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", peaks),
            ("start_indices", starts),
            ("end_indices", ends),
        )
        reference = (
            None
            if reference_signal is None
            else _require_1d_array("reference_signal", reference_signal)
        )
        valid, percentages, reference_values = percentage_under_baseline(
            array,
            fs,
            peaks,
            starts,
            ends,
            baseline_array,
            aub_window_s=aub_window_samples,
            ref_signal=reference,
            aub_threshold=aub_threshold,
        )
        return np.asarray(valid), np.asarray(percentages), np.asarray(reference_values)

    def detect_local_high_aub(
        self,
        aub_values: Any,
        *,
        threshold_percentile: float = 75.0,
        threshold_factor: float = 4.0,
    ) -> np.ndarray:
        """Flag area-under-baseline values that are locally high outliers."""

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_local_high_aub,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        values = _require_1d_array("aub_values", aub_values)
        _require_percentile("threshold_percentile", threshold_percentile)
        return np.asarray(
            detect_local_high_aub(
                values,
                threshold_percentile=threshold_percentile,
                threshold_factor=threshold_factor,
            )
        )

    def detect_extreme_time_products(
        self,
        time_products: Any,
        *,
        upper_percentile: float = 95.0,
        upper_factor: float = 10.0,
        lower_percentile: float = 5.0,
        lower_factor: float = 0.1,
    ) -> np.ndarray:
        """Flag time-product (area above baseline) values outside the
        expected percentile-based bounds."""

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_extreme_time_products,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        values = _require_1d_array("time_products", time_products)
        _require_percentile("upper_percentile", upper_percentile)
        _require_percentile("lower_percentile", lower_percentile)
        return np.asarray(
            detect_extreme_time_products(
                values,
                upper_percentile=upper_percentile,
                upper_factor=upper_factor,
                lower_percentile=lower_percentile,
                lower_factor=lower_factor,
            )
        )

    def detect_non_consecutive_manoeuvres(
        self,
        ventilator_breath_indices: Any,
        manoeuvre_indices: Any,
    ) -> np.ndarray:
        """Flag manoeuvres (e.g. Pocc) with no supported ventilator breath
        between them and the next manoeuvre."""

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_non_consecutive_manoeuvres,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        breaths = _require_index_array(
            "ventilator_breath_indices", ventilator_breath_indices
        )
        manoeuvres = _require_index_array("manoeuvre_indices", manoeuvre_indices)
        return np.asarray(detect_non_consecutive_manoeuvres(breaths, manoeuvres))

    def evaluate_bell_curve_error(
        self,
        peak_indices: Any,
        start_indices: Any,
        end_indices: Any,
        signal: Any,
        time_products: Any,
        *,
        sample_frequency: float,
        bell_window_samples: int | None = None,
        bell_threshold: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """How well each breath's envelope fits a bell curve
        (Warnaar et al. 2024).

        Returns `(valid_peak, percentage_bell_error, bell_error, y_min,
        fitted_parameters)`.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_bell_curve_error,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        peaks = _require_index_array("peak_indices", peak_indices)
        starts = _require_index_array("start_indices", start_indices)
        ends = _require_index_array("end_indices", end_indices)
        array = _require_1d_array("signal", signal)
        products = _require_1d_array("time_products", time_products)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", peaks),
            ("start_indices", starts),
            ("end_indices", ends),
            ("time_products", products),
        )
        (
            valid_peak,
            percentage_bell_error,
            bell_error,
            y_min,
            fitted_parameters,
        ) = evaluate_bell_curve_error(
            peaks,
            starts,
            ends,
            array,
            fs,
            products,
            bell_window_s=bell_window_samples,
            bell_threshold=bell_threshold,
        )
        return (
            np.asarray(valid_peak),
            np.asarray(percentage_bell_error),
            np.asarray(bell_error),
            np.asarray(y_min),
            np.asarray(fitted_parameters),
        )

    def evaluate_event_timing(
        self,
        first_event_times: Any,
        second_event_times: Any,
        *,
        min_delta: float = 0.0,
        max_delta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whether `first_event_times` precede the paired `second_event_times`
        by between `min_delta` and `max_delta` seconds.

        Returns `(correct_timing, delta_time)`.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_event_timing,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        first = _require_1d_array("first_event_times", first_event_times)
        second = _require_1d_array("second_event_times", second_event_times)
        _require_equal_length(
            ("first_event_times", first), ("second_event_times", second)
        )
        correct_timing, delta_time = evaluate_event_timing(
            first,
            second,
            delta_min=min_delta,
            delta_max=max_delta,
        )
        return np.asarray(correct_timing), np.asarray(delta_time)

    def evaluate_respiratory_rates(
        self,
        emg_breath_indices: Any,
        recording_duration_seconds: float,
        ventilator_respiratory_rate: float,
        *,
        minimum_fraction: float = 0.1,
    ) -> tuple[float, bool]:
        """Fraction of expected EMG breaths actually detected, relative to
        the ventilator's respiratory rate.

        Returns `(detected_fraction, criterion_met)`.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_respiratory_rates,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        breaths = _require_index_array("emg_breath_indices", emg_breath_indices)
        _require_finite_positive(
            "recording_duration_seconds", recording_duration_seconds
        )
        _require_finite_positive(
            "ventilator_respiratory_rate", ventilator_respiratory_rate
        )
        detected_fraction, criterion_met = evaluate_respiratory_rates(
            breaths,
            recording_duration_seconds,
            ventilator_respiratory_rate,
            min_fraction=minimum_fraction,
        )
        return float(detected_fraction), bool(criterion_met)

    def _preprocess_default(
        self,
        recording: Any,
        *,
        channel: int = 0,
        high_pass_hz: float = 80,
        low_pass_hz: float | None = None,
        envelope_window_seconds: float = 0.5,
        notch_base_frequency: float | None = None,
        notch_max_frequency: float | None = None,
        notch_quality_factor: float = 30.0,
    ) -> dict[str, Any]:
        """Run the Stage 1 EMG preprocessing pipeline through ReSurfEMG.

        ``notch_base_frequency`` opts into harmonic notch filtering (e.g.
        ``50.0`` for mains hum, or a co-recorded EIT device's frame rate, which
        injects a harmonic comb into the sEMG whenever the EIT device is
        running simultaneously). It is applied to the band-passed signal, after
        ``emg_bandpass_butter`` and before the envelope is computed, so a
        narrow high-pass alone (which only removes the fundamental) doesn't
        leave higher harmonics inside the pass band untouched.
        """

        try:
            import numpy as np
            from resurfemg.preprocessing.filtering import emg_bandpass_butter
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG preprocessing requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"`.'
            ) from exc

        _require_emg_recording(recording)

        metadata = dict(recording["metadata"])
        fs = float(metadata["fs"])
        array = recording["array"]
        raw = np.asarray(array[channel], dtype=float)

        if low_pass_hz is None:
            low_pass_hz = min(fs / 2 * 0.95, 500)

        filtered = emg_bandpass_butter(
            emg_raw=raw,
            high_pass=high_pass_hz,
            low_pass=low_pass_hz,
            fs_emg=fs,
        )
        if notch_base_frequency is not None:
            # Default the notch's reach to Nyquist, not `low_pass_hz`: a
            # harmonic landing at or just past the low-pass cutoff (e.g. the
            # EIT frame-rate comb's 10th harmonic sitting on a 500 Hz
            # low-pass edge) is only partially attenuated by the low-pass
            # filter's finite roll-off, so it must still be fully inside the
            # notch's stopband rather than at its boundary.
            filtered = harmonic_notch_filter(
                filtered,
                base_frequency=notch_base_frequency,
                sample_frequency=fs,
                max_frequency=notch_max_frequency or (fs / 2),
                quality_factor=notch_quality_factor,
            )
        envelope_window_samples = max(1, int(envelope_window_seconds * fs))
        envelope = rolling_arv(filtered, window_length=envelope_window_samples)

        return {
            **recording,
            "channel": channel,
            "fs": fs,
            "raw_channel": raw,
            "filtered": filtered,
            "envelope": envelope,
            "filter": {
                "high_pass_hz": high_pass_hz,
                "low_pass_hz": low_pass_hz,
                "envelope_window_seconds": envelope_window_seconds,
                "notch_base_frequency": notch_base_frequency,
                "notch_max_frequency": (
                    (notch_max_frequency or (fs / 2))
                    if notch_base_frequency is not None
                    else None
                ),
                "notch_quality_factor": (
                    notch_quality_factor if notch_base_frequency is not None else None
                ),
            },
        }

    def _detect_breaths_default(
        self,
        processed_emg: Any,
        *,
        min_breath_width_seconds: float = 1.0,
        half_window_seconds: float = 0.5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run ReSurfEMG EMG breath detection and return common rows."""

        if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
            raise UnsupportedWorkflowError(
                "Default EMG breath detection expects processed EMG data from "
                "`preprocess_emg()`. Pass `detector=callable` to normalize "
                "custom detections."
            )

        fs = float(processed_emg["fs"])
        envelope = processed_emg["envelope"]
        min_width_samples = max(1, int(min_breath_width_seconds * fs))
        half_window_samples = max(1, int(half_window_seconds * fs))

        peak_indices = detect_emg_breath_peaks(
            envelope,
            min_peak_width_samples=min_width_samples,
            **kwargs,
        )

        events = []
        for peak_index in peak_indices:
            start_index = max(0, int(peak_index) - half_window_samples)
            end_index = min(len(envelope) - 1, int(peak_index) + half_window_samples)
            events.append(
                {
                    "start_time": start_index / fs,
                    "end_time": end_index / fs,
                    "peak_time": int(peak_index) / fs,
                    "start_index": start_index,
                    "peak_index": int(peak_index),
                    "end_index": end_index,
                    "sample_frequency": fs,
                    "signal_name": processed_emg["channel"],
                    "source": "resurfemg.detect_emg_breaths",
                }
            )

        return events

    def _postprocess_default(
        self,
        processed_emg: Any,
        events: Sequence[BreathEvent] | None = None,
        *,
        ventilator: Any | None = None,
        ventilator_pressure_channel: int = 0,
        ventilator_flow_channel: int = 1,
        ventilator_volume_channel: int = 2,
        ventilator_fs: float | None = None,
        ventilator_breath_width_seconds: float = 0.5,
        peep: float | None = None,
        baseline_window_seconds: float = 30.0,
        baseline_step_seconds: float = 1.0,
        baseline_percentile: float = 33.0,
        slope_window_seconds: float = 0.5,
        aub_window_seconds: float = 5.0,
        selected_functions: dict[str, dict[str, bool]] | None = None,
    ) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise OptionalDependencyError("EMG postprocessing requires numpy.") from exc

        if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
            raise UnsupportedWorkflowError(
                "Default EMG postprocessing expects processed EMG data from "
                "`preprocess_emg()`."
            )

        envelope = np.asarray(processed_emg["envelope"], dtype=float)
        fs = float(processed_emg["fs"])
        window_samples = max(1, int(baseline_window_seconds * fs))
        step_samples = max(1, int(baseline_step_seconds * fs))
        selected = _normalize_selected_postprocessing(selected_functions)
        enabled = selected.__contains__

        computed: dict[str, Any] = {
            "baseline": {},
            "event_detection": {},
            "features": {},
            "quality_assessment": {},
        }
        skipped: dict[str, str] = {}

        peak_indices = peak_indices_from_events(events, fs)
        peak_indices_array = np.asarray(peak_indices, dtype=int)
        unavailable_reason = _missing_postprocessing_dependency()
        if unavailable_reason is not None:
            return _unavailable_postprocessing_result(
                selected=selected,
                peak_indices=peak_indices_array,
                computed=computed,
                reason=unavailable_reason,
                settings={
                    "baseline_window_seconds": baseline_window_seconds,
                    "baseline_step_seconds": baseline_step_seconds,
                    "baseline_percentile": baseline_percentile,
                    "slope_window_seconds": slope_window_seconds,
                    "aub_window_seconds": aub_window_seconds,
                    "ventilator_breath_width_seconds": ventilator_breath_width_seconds,
                    "peep": peep,
                },
            )

        baseline = None
        if enabled(("baseline", "moving_baseline")):
            moving_baseline = self.moving_baseline(
                envelope,
                window_samples=window_samples,
                step_samples=step_samples,
                percentile=baseline_percentile,
            )
            computed["baseline"]["moving_baseline"] = moving_baseline
            baseline = moving_baseline

        if enabled(("baseline", "slopesum_baseline")):
            slopesum_baseline = self.slopesum_baseline(
                envelope,
                window_samples=window_samples,
                step_samples=step_samples,
                sample_frequency=fs,
                percentile=baseline_percentile,
                moving_average_samples=max(1, int(fs // 2)),
                percentile_window_samples=max(1, int(fs)),
            )
            computed["baseline"]["slopesum_baseline"] = {
                "baseline": slopesum_baseline[0],
                "running_mean": slopesum_baseline[1],
                "running_std": slopesum_baseline[2],
                "series": slopesum_baseline[3],
            }
            baseline = slopesum_baseline[0]

        vent_signals = ventilator_signals(
            ventilator,
            pressure_channel=ventilator_pressure_channel,
            flow_channel=ventilator_flow_channel,
            volume_channel=ventilator_volume_channel,
            fs=ventilator_fs,
        )

        ventilator_breath_indices = np.asarray([], dtype=int)
        if vent_signals is not None:
            v_vent = vent_signals["volume"]
            p_vent = vent_signals["pressure"]
            vent_fs = float(vent_signals["fs"])
            vent_width_samples = max(1, int(ventilator_breath_width_seconds * vent_fs))
            if enabled(("event_detection", "detect_ventilator_breath")):
                ventilator_breath_indices = np.asarray(
                    detect_ventilator_breath_peaks(
                        v_vent,
                        start_index=0,
                        end_index=len(v_vent) - 1,
                        width_samples=vent_width_samples,
                    ),
                    dtype=int,
                )
                computed["event_detection"]["detect_ventilator_breath"] = (
                    ventilator_breath_indices
                )

            pocc_indices = np.asarray([], dtype=int)
            if enabled(("event_detection", "find_occluded_breaths")):
                if peep is None:
                    peep = float(np.nanmedian(p_vent))
                pocc_indices = np.asarray(
                    detect_occluded_breath_peaks(
                        p_vent,
                        sample_frequency=vent_fs,
                        peep=peep,
                    ),
                    dtype=int,
                )
                computed["event_detection"]["find_occluded_breaths"] = pocc_indices

            if enabled(("quality_assessment", "detect_non_consecutive_manoeuvres")):
                if len(ventilator_breath_indices) and len(pocc_indices):
                    computed["quality_assessment"][
                        "detect_non_consecutive_manoeuvres"
                    ] = self.detect_non_consecutive_manoeuvres(
                        ventilator_breath_indices,
                        pocc_indices,
                    )
                else:
                    skipped["quality_assessment.detect_non_consecutive_manoeuvres"] = (
                        "Needs ventilator breath and manoeuvre indices."
                    )

            if (
                enabled(("quality_assessment", "evaluate_respiratory_rates"))
                and len(ventilator_breath_indices) >= 2
            ):
                computed["quality_assessment"]["ventilator_respiratory_rate"] = (
                    self.run_postprocessing_function(
                        "features",
                        "respiratory_rate",
                        ventilator_breath_indices,
                        vent_fs,
                    )
                )
            else:
                skipped["quality_assessment.ventilator_respiratory_rate"] = (
                    "Needs at least two ventilator breaths."
                )
        else:
            skipped.update(
                {
                    "event_detection.find_occluded_breaths": "Needs ventilator pressure and PEEP inputs.",
                    "event_detection.detect_ventilator_breath": "Needs ventilator volume input.",
                    "quality_assessment.detect_non_consecutive_manoeuvres": "Needs ventilator breath and manoeuvre indices.",
                    "quality_assessment.ventilator_respiratory_rate": "Needs ventilator volume input.",
                }
            )

        start_indices = None
        end_indices = None
        if len(peak_indices_array) and baseline is not None:
            if enabled(("event_detection", "onoffpeak_baseline_crossing")):
                computed["event_detection"]["onoffpeak_baseline_crossing"] = (
                    onoff_from_baseline_crossings(
                        envelope,
                        baseline,
                        peak_indices_array,
                    )
                )
                start_indices, end_indices, *_ = computed["event_detection"][
                    "onoffpeak_baseline_crossing"
                ]
            slope_window_samples = max(1, int(slope_window_seconds * fs))
            if enabled(("event_detection", "onoffpeak_slope_extrapolation")):
                computed["event_detection"]["onoffpeak_slope_extrapolation"] = (
                    onoff_from_slope(
                        envelope,
                        sample_frequency=fs,
                        peak_indices=peak_indices_array,
                        slope_window=slope_window_samples,
                    )
                )

            if start_indices is not None and end_indices is not None:
                if enabled(("features", "time_to_peak")):
                    computed["features"]["time_to_peak"] = time_to_peak(
                        envelope,
                        start_indices,
                        end_indices,
                    )
                if enabled(("features", "pseudo_slope")):
                    computed["features"]["pseudo_slope"] = pseudo_slope(
                        envelope,
                        start_indices,
                        end_indices,
                    )
                if enabled(("features", "amplitude")):
                    computed["features"]["amplitude"] = amplitude_at_peaks(
                        envelope,
                        peak_indices_array,
                        baseline,
                    )
                if enabled(("features", "time_product")):
                    computed["features"]["time_product"] = window_integral(
                        envelope,
                        fs,
                        start_indices,
                        end_indices,
                        baseline,
                    )
                if enabled(("features", "area_under_baseline")):
                    computed["features"]["area_under_baseline"] = area_under_baseline(
                        envelope,
                        fs,
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        max(1, int(aub_window_seconds * fs)),
                        baseline,
                    )
            else:
                skipped["features"] = (
                    "Needs event_detection.onoffpeak_baseline_crossing."
                )

            if (
                enabled(("features", "respiratory_rate"))
                and len(peak_indices_array) >= 2
            ):
                computed["features"]["respiratory_rate"] = (
                    respiratory_rate_from_indices(peak_indices_array, fs)
                )
            elif enabled(("features", "respiratory_rate")):
                skipped["features.respiratory_rate"] = "Needs at least two EMG breaths."

            if enabled(("quality_assessment", "snr_pseudo")):
                computed["quality_assessment"]["snr_pseudo"] = self.snr_pseudo(
                    envelope,
                    peak_indices_array,
                    baseline,
                    sample_frequency=fs,
                )
            if (
                enabled(("quality_assessment", "percentage_under_baseline"))
                and start_indices is not None
                and end_indices is not None
            ):
                computed["quality_assessment"]["percentage_under_baseline"] = (
                    self.percentage_under_baseline(
                        envelope,
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        baseline,
                        sample_frequency=fs,
                    )
                )
            if (
                enabled(("quality_assessment", "detect_local_high_aub"))
                and "area_under_baseline" in computed["features"]
            ):
                aubs = computed["features"]["area_under_baseline"][0]
                computed["quality_assessment"]["detect_local_high_aub"] = (
                    self.detect_local_high_aub(aubs)
                )
            if (
                enabled(("quality_assessment", "detect_extreme_time_products"))
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["detect_extreme_time_products"] = (
                    self.detect_extreme_time_products(time_products)
                )
            if (
                enabled(("quality_assessment", "evaluate_bell_curve_error"))
                and start_indices is not None
                and end_indices is not None
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["evaluate_bell_curve_error"] = (
                    self.evaluate_bell_curve_error(
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        envelope,
                        time_products,
                        sample_frequency=fs,
                    )
                )
            if (
                enabled(("quality_assessment", "evaluate_event_timing"))
                and vent_signals is not None
                and len(ventilator_breath_indices)
            ):
                paired_count = min(
                    len(peak_indices_array), len(ventilator_breath_indices)
                )
                computed["quality_assessment"]["evaluate_event_timing"] = (
                    self.evaluate_event_timing(
                        peak_indices_array[:paired_count] / fs,
                        ventilator_breath_indices[:paired_count]
                        / float(vent_signals["fs"]),
                    )
                )
                if (
                    enabled(("quality_assessment", "evaluate_respiratory_rates"))
                    and len(ventilator_breath_indices) >= 2
                ):
                    rr_vent = computed["quality_assessment"][
                        "ventilator_respiratory_rate"
                    ][0]
                    computed["quality_assessment"]["evaluate_respiratory_rates"] = (
                        self.evaluate_respiratory_rates(
                            peak_indices_array,
                            len(envelope) / fs,
                            rr_vent,
                        )
                    )
                elif enabled(("quality_assessment", "evaluate_respiratory_rates")):
                    skipped["quality_assessment.evaluate_respiratory_rates"] = (
                        "Needs at least two ventilator breaths."
                    )
            elif enabled(("quality_assessment", "evaluate_event_timing")):
                skipped["quality_assessment.evaluate_event_timing"] = (
                    "Needs ventilator breath timing."
                )
                if enabled(("quality_assessment", "evaluate_respiratory_rates")):
                    skipped["quality_assessment.evaluate_respiratory_rates"] = (
                        "Needs ventilator respiratory rate."
                    )
        else:
            for name in (
                "onoffpeak_baseline_crossing",
                "onoffpeak_slope_extrapolation",
                "time_to_peak",
                "pseudo_slope",
                "amplitude",
                "time_product",
                "area_under_baseline",
                "respiratory_rate",
                "snr_pseudo",
                "percentage_under_baseline",
                "detect_local_high_aub",
                "detect_extreme_time_products",
                "evaluate_bell_curve_error",
            ):
                category = _category_for_function(name)
                if category is not None and enabled((category, name)):
                    skipped[f"{category}.{name}"] = (
                        "Needs detected EMG breath peaks and baseline."
                    )
            if enabled(("quality_assessment", "evaluate_event_timing")):
                skipped["quality_assessment.evaluate_event_timing"] = (
                    "Needs detected EMG breath peaks."
                )
            if enabled(("quality_assessment", "evaluate_respiratory_rates")):
                skipped["quality_assessment.evaluate_respiratory_rates"] = (
                    "Needs detected EMG breath peaks."
                )

        if enabled(("quality_assessment", "pocc_quality")):
            skipped["quality_assessment.pocc_quality"] = "Needs Pocc ventilator inputs."
        if enabled(("quality_assessment", "interpeak_dist")):
            skipped["quality_assessment.interpeak_dist"] = "Needs ECG peak indices."

        return {
            "available": self.available_postprocessing(),
            "computed": computed,
            "skipped": skipped,
            "peak_indices": peak_indices_array,
            "settings": {
                "baseline_window_seconds": baseline_window_seconds,
                "baseline_step_seconds": baseline_step_seconds,
                "baseline_percentile": baseline_percentile,
                "slope_window_seconds": slope_window_seconds,
                "aub_window_seconds": aub_window_seconds,
                "ventilator_breath_width_seconds": ventilator_breath_width_seconds,
                "peep": peep,
            },
        }


def ventilator_signals(
    ventilator: Any | None,
    *,
    pressure_channel: int,
    flow_channel: int,
    volume_channel: int,
    fs: float | None = None,
) -> dict[str, Any] | None:
    if ventilator is None:
        return None

    try:
        import numpy as np
    except ImportError as exc:
        raise OptionalDependencyError("EMG postprocessing requires numpy.") from exc

    metadata = ventilator.get("metadata", {}) if isinstance(ventilator, dict) else {}
    array = ventilator.get("array") if isinstance(ventilator, dict) else ventilator
    if array is None:
        raise TypeError("Ventilator postprocessing input needs an array.")

    vent_fs = fs if fs is not None else metadata.get("fs")
    if vent_fs is None:
        raise TypeError("Ventilator postprocessing input needs a sampling rate.")

    array = np.asarray(array, dtype=float)
    return {
        "pressure": np.asarray(array[pressure_channel], dtype=float),
        "flow": np.asarray(array[flow_channel], dtype=float),
        "volume": np.asarray(array[volume_channel], dtype=float),
        "fs": float(vent_fs),
        "metadata": metadata,
    }


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


def peak_indices_from_events(
    events: Sequence[BreathEvent] | None, fs: float
) -> list[int]:
    if events is None:
        return []
    return [
        int(event.peak_time * fs) for event in events if event.peak_time is not None
    ]


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
