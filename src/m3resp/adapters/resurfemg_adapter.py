"""Adapter boundary for the upstream `resurfemg` package."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.events import coerce_breath_events
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.signals import ProcessingState

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


class ReSurfEMGAdapter:
    """Thin wrapper around `resurfemg`."""

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, path: str, **kwargs: Any) -> Any:
        """Load EMG data through `resurfemg` or an injected loader."""

        if self._loader is not None:
            return self._loader(path, **kwargs)

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
            QualityFlag(
                name=name,
                passed=_quality_result_passed(value),
                severity="info",
                modality="emg",
                value=_scalar_or_none(value),
            )
            for name, value in quality_results.items()
        ]
        for name, reason in postprocessed.get("skipped", {}).items():
            flags.append(
                QualityFlag(
                    name=name,
                    passed=False,
                    severity="warning",
                    modality="emg",
                    message=str(reason),
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

    def _preprocess_default(
        self,
        recording: Any,
        *,
        channel: int = 0,
        high_pass_hz: float = 80,
        low_pass_hz: float | None = None,
        envelope_window_seconds: float = 0.5,
    ) -> dict[str, Any]:
        """Run the Stage 1 EMG preprocessing pipeline through ReSurfEMG."""

        try:
            import numpy as np
            from resurfemg.preprocessing.envelope import full_rolling_arv
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
        envelope_window_samples = max(1, int(envelope_window_seconds * fs))
        envelope = full_rolling_arv(filtered, envelope_window_samples)

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

        try:
            from resurfemg.postprocessing.event_detection import detect_emg_breaths
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG breath detection requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"`.'
            ) from exc

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

        peak_indices = detect_emg_breaths(
            emg_env=envelope,
            min_peak_width_s=min_width_samples,
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
                    "source": "resurfemg.detect_emg_breaths",
                    "metadata": {
                        "start_index": start_index,
                        "peak_index": int(peak_index),
                        "end_index": end_index,
                        "channel": processed_emg["channel"],
                    },
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

        peak_indices = _peak_indices_from_events(events, fs)
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
            moving_baseline = self.run_postprocessing_function(
                "baseline",
                "moving_baseline",
                envelope,
                window_samples,
                step_samples,
                set_percentile=baseline_percentile,
            )
            computed["baseline"]["moving_baseline"] = moving_baseline
            baseline = moving_baseline

        if enabled(("baseline", "slopesum_baseline")):
            slopesum_baseline = self.run_postprocessing_function(
                "baseline",
                "slopesum_baseline",
                envelope,
                window_samples,
                step_samples,
                fs,
                set_percentile=baseline_percentile,
                ma_window=max(1, int(fs // 2)),
                perc_window=max(1, int(fs)),
            )
            computed["baseline"]["slopesum_baseline"] = {
                "baseline": slopesum_baseline[0],
                "running_mean": slopesum_baseline[1],
                "running_std": slopesum_baseline[2],
                "series": slopesum_baseline[3],
            }
            baseline = slopesum_baseline[0]

        ventilator_signals = _ventilator_signals(
            ventilator,
            pressure_channel=ventilator_pressure_channel,
            flow_channel=ventilator_flow_channel,
            volume_channel=ventilator_volume_channel,
            fs=ventilator_fs,
        )

        ventilator_breath_indices = np.asarray([], dtype=int)
        if ventilator_signals is not None:
            v_vent = ventilator_signals["volume"]
            p_vent = ventilator_signals["pressure"]
            vent_fs = float(ventilator_signals["fs"])
            vent_width_samples = max(1, int(ventilator_breath_width_seconds * vent_fs))
            if enabled(("event_detection", "detect_ventilator_breath")):
                ventilator_breath_indices = np.asarray(
                    self.run_postprocessing_function(
                        "event_detection",
                        "detect_ventilator_breath",
                        v_vent,
                        0,
                        len(v_vent) - 1,
                        vent_width_samples,
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
                    self.run_postprocessing_function(
                        "event_detection",
                        "find_occluded_breaths",
                        p_vent,
                        vent_fs,
                        peep,
                    ),
                    dtype=int,
                )
                computed["event_detection"]["find_occluded_breaths"] = pocc_indices

            if enabled(("quality_assessment", "detect_non_consecutive_manoeuvres")):
                if len(ventilator_breath_indices) and len(pocc_indices):
                    computed["quality_assessment"][
                        "detect_non_consecutive_manoeuvres"
                    ] = self.run_postprocessing_function(
                        "quality_assessment",
                        "detect_non_consecutive_manoeuvres",
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
                    self.run_postprocessing_function(
                        "event_detection",
                        "onoffpeak_baseline_crossing",
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
                    self.run_postprocessing_function(
                        "event_detection",
                        "onoffpeak_slope_extrapolation",
                        envelope,
                        fs,
                        peak_indices_array,
                        slope_window_samples,
                    )
                )

            if start_indices is not None and end_indices is not None:
                if enabled(("features", "time_to_peak")):
                    computed["features"]["time_to_peak"] = (
                        self.run_postprocessing_function(
                            "features",
                            "time_to_peak",
                            envelope,
                            start_indices,
                            end_indices,
                        )
                    )
                if enabled(("features", "pseudo_slope")):
                    computed["features"]["pseudo_slope"] = (
                        self.run_postprocessing_function(
                            "features",
                            "pseudo_slope",
                            envelope,
                            start_indices,
                            end_indices,
                        )
                    )
                if enabled(("features", "amplitude")):
                    computed["features"]["amplitude"] = (
                        self.run_postprocessing_function(
                            "features",
                            "amplitude",
                            envelope,
                            peak_indices_array,
                            baseline,
                        )
                    )
                if enabled(("features", "time_product")):
                    computed["features"]["time_product"] = (
                        self.run_postprocessing_function(
                            "features",
                            "time_product",
                            envelope,
                            fs,
                            start_indices,
                            end_indices,
                            baseline,
                        )
                    )
                if enabled(("features", "area_under_baseline")):
                    computed["features"]["area_under_baseline"] = (
                        self.run_postprocessing_function(
                            "features",
                            "area_under_baseline",
                            envelope,
                            fs,
                            peak_indices_array,
                            start_indices,
                            end_indices,
                            max(1, int(aub_window_seconds * fs)),
                            baseline,
                        )
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
                    self.run_postprocessing_function(
                        "features", "respiratory_rate", peak_indices_array, fs
                    )
                )
            elif enabled(("features", "respiratory_rate")):
                skipped["features.respiratory_rate"] = "Needs at least two EMG breaths."

            if enabled(("quality_assessment", "snr_pseudo")):
                computed["quality_assessment"]["snr_pseudo"] = (
                    self.run_postprocessing_function(
                        "quality_assessment",
                        "snr_pseudo",
                        envelope,
                        peak_indices_array,
                        baseline,
                        fs,
                    )
                )
            if (
                enabled(("quality_assessment", "percentage_under_baseline"))
                and start_indices is not None
                and end_indices is not None
            ):
                computed["quality_assessment"]["percentage_under_baseline"] = (
                    self.run_postprocessing_function(
                        "quality_assessment",
                        "percentage_under_baseline",
                        envelope,
                        fs,
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        baseline,
                    )
                )
            if (
                enabled(("quality_assessment", "detect_local_high_aub"))
                and "area_under_baseline" in computed["features"]
            ):
                aubs = computed["features"]["area_under_baseline"][0]
                computed["quality_assessment"]["detect_local_high_aub"] = (
                    self.run_postprocessing_function(
                        "quality_assessment", "detect_local_high_aub", aubs
                    )
                )
            if (
                enabled(("quality_assessment", "detect_extreme_time_products"))
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["detect_extreme_time_products"] = (
                    self.run_postprocessing_function(
                        "quality_assessment",
                        "detect_extreme_time_products",
                        time_products,
                    )
                )
            if (
                enabled(("quality_assessment", "evaluate_bell_curve_error"))
                and start_indices is not None
                and end_indices is not None
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["evaluate_bell_curve_error"] = (
                    self.run_postprocessing_function(
                        "quality_assessment",
                        "evaluate_bell_curve_error",
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        envelope,
                        fs,
                        time_products,
                    )
                )
            if (
                enabled(("quality_assessment", "evaluate_event_timing"))
                and ventilator_signals is not None
                and len(ventilator_breath_indices)
            ):
                paired_count = min(
                    len(peak_indices_array), len(ventilator_breath_indices)
                )
                computed["quality_assessment"]["evaluate_event_timing"] = (
                    self.run_postprocessing_function(
                        "quality_assessment",
                        "evaluate_event_timing",
                        peak_indices_array[:paired_count] / fs,
                        ventilator_breath_indices[:paired_count]
                        / float(ventilator_signals["fs"]),
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
                        self.run_postprocessing_function(
                            "quality_assessment",
                            "evaluate_respiratory_rates",
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


def _ventilator_signals(
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


def _quality_result_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    array = np.asarray(value)
    if array.dtype == bool:
        return bool(array.all())
    return True


def _scalar_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _peak_indices_from_events(
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
