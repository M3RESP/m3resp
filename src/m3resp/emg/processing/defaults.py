"""Native fallback implementations of `ReSurfEMGAdapter`, used when the optional `resurfemg` package is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.processing.filters import bandpass_filter, harmonic_notch_filter
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
from m3resp.processing.windows import rolling_envelope

from .baseline import (
    moving_baseline as compute_moving_baseline,
)
from .baseline import (
    slopesum_baseline as compute_slopesum_baseline,
)
from .ecg import detect_ecg_peaks, gate_ecg, wavelet_denoise_ecg
from .quality import (
    detect_extreme_time_products,
    detect_local_high_aub,
    detect_non_consecutive_manoeuvres,
    evaluate_bell_curve_error,
    evaluate_event_timing,
    evaluate_respiratory_rates,
    percentage_under_baseline,
    snr_pseudo,
)
from .shared import (
    _category_for_function,
    _choose_emg_channel,
    _mask_invalid,
    _normalize_selected_postprocessing,
    _require_emg_recording,
)
from .signals import peak_indices_from_events, ventilator_signals

if TYPE_CHECKING:
    from collections.abc import Sequence

    from m3resp.data.events import BreathEvent

BASELINE_FUNCTIONS = {
    "baseline": {
        "moving_baseline": True,
        "slopesum_baseline": True,
    }
}

EVENT_DETECTION_FUNCTIONS = {
    "event_detection": {
        "onoffpeak_baseline_crossing": True,
        "onoffpeak_slope_extrapolation": True,
    }
}
VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS = {
    "quality_assessment": {
        "detect_non_consecutive_manoeuvres": True,
        "evaluate_respiratory_rates": True,
    }
}
MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS = {
    "quality_assessment": {
        "evaluate_event_timing": True,
        "evaluate_respiratory_rates": True,
        "ventilator_respiratory_rate": True,
    }
}

EMG_QUALITY_ASSESSMENT_FUNCTIONS = {
    "quality_assessment": {
        "snr_pseudo": True,
        "percentage_under_baseline": True,
        "detect_local_high_aub": True,
        "detect_extreme_time_products": True,
        "evaluate_bell_curve_error": True,
        "evaluate_event_timing": True,
        "evaluate_respiratory_rates": True,
    }
}

FEATURES_FUNCTIONS = {
    "features": {
        "amplitude_at_peaks": True,
        "area_under_baseline": True,
        "pseudo_slope": True,
        "time_to_peak": True,
        "window_integral": True,
    }
}


def default_low_pass_hz(sample_frequency: float) -> float:
    """Default EMG low-pass cutoff in Hz: 500 Hz, or 95 % of Nyquist if lower."""

    return min(sample_frequency / 2 * 0.95, 500)


def notch_max_frequency_hz(
    sample_frequency: float, max_frequency: float | None
) -> float:
    """Highest harmonic the notch filter removes, in Hz.

    None or zero gives the Nyquist frequency (half the sampling rate). A value
    above the Nyquist frequency is lowered to it, with a warning, because the
    filter removes no harmonics above it.
    """

    nyquist = sample_frequency / 2
    if not max_frequency:
        return nyquist
    if max_frequency > nyquist:
        warnings.warn(
            f"Notch filter max_frequency ({max_frequency} Hz) is above the "
            f"Nyquist frequency ({nyquist} Hz for a sampling rate of "
            f"{sample_frequency} Hz); harmonics are removed up to {nyquist} Hz.",
            UserWarning,
            stacklevel=3,
        )
        return nyquist
    return max_frequency


def _preprocess_default(
    recording: Any,
    *,
    channel: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run a default EMG preprocessing pipeline and return a dictionary of results.

    Args:
        recording: the raw emg bundle
        channel (int, Optional): the channel index to process.
            If None, the first channel is used.
        **kwargs: optional keyword arguments for preprocessing customization

    Returns:
        dict[str, Any]: a dictionary containing the processed EMG data and metadata
            - "channel": the channel index used for processing
            - "fs": the sampling frequency of the EMG signal
            - "raw_channel": the raw EMG signal for the selected channel
            - "filtered": the band-pass filtered EMG signal
            - "clean": the ECG-cleaned EMG signal (or the filtered signal if ECG
                removal is disabled)
            - "envelope": the computed envelope of the EMG signal (or None if
                envelope computation is disabled)
            - "filter": a dictionary containing the filter parameters used for
                preprocessing

    Without any additional arguments, the function applies the ReSurfEMG default
    preprocessing pipeline:
    -- band-pass filter (20-500 Hz)
    -- ecg peak detection and removal via gating
    -- RMS envelope computation

    # Preprocessing pipeline customization
    ## Filtering
    filtering is performed via 3rd order Butterworth filtering with
    customizable pass band:
    - ``high_pass_hz``: high-pass cutoff frequency in Hz (default: 20.0)
    - ``low_pass_hz``: low-pass cutoff frequency in Hz
        (default: min(fs / 2 * 0.95, 500))

    An optional harmonic notch filter can be applied to the band-passed signal,
    with customizable parameters:
    - ``notch_base_frequency``: base frequency of the notch filter in Hz
        (default: None)
    - ``notch_max_frequency``: maximum frequency of the notch filter in Hz
        (default: fs / 2)
    - ``notch_quality_factor``: quality factor of the notch filter
        (default: 30.0 if notch_base_frequency is not None, else None)
    - ``notch_before_bandpass``: whether to apply the notch filter before the
        band-pass filter
        (default: False if notch_base_frequency is not None, else None)

    ## ECG removal
    ECG removal is an optional step that is applied by default.
    - ``remove_ecg``: whether to remove ECG from the signal
        (default: True)
    - ``ecg_removal_method``: method for ECG removal, either "gating" or "wavelet"
        (default: "gating")
    - ``ecg_gate_fill_method``: method for filling gated ECG segments:
        - 0: fill with zeros
        - 1: fill with linear interpolation (default)
        - 2: fill with the average of the surrounding signal segments
        - 3: fill with the running average of the RMS

    ## Envelope computation
    The envelope is computed by default, but can be disabled
    - ``compute_envelope``: whether to compute the envelope
        (default: True)
    - ``envelope_method``: method for envelope computation, either "rms" or "arv"
        (default: "rms")
    - ``envelope_window_seconds``: window length for envelope computation in seconds
        (default: 0.25)

    # Additional information
    The band-pass defaults to 20-500 Hz, the range respiratory-sEMG
    literature specifies. The high-pass is deliberately *not* set low
    enough to double as ECG suppression: removing ECG is the job of a
    dedicated gating step (``emg.ecg_gating``, which the ``"emg"`` preset
    runs by default), because a high-pass steep enough to attenuate the
    QRS complex still leaves its higher-frequency content inside the pass
    band.

    ``notch_base_frequency`` opts into harmonic notch filtering (e.g.
    ``50.0`` for mains hum, or a co-recorded EIT device's frame rate, which
    injects a harmonic comb into the sEMG whenever the EIT device is
    running simultaneously). It is applied to the band-passed signal, after
    ``emg_bandpass_butter`` and before the envelope is computed, so a
    narrow high-pass alone (which only removes the fundamental) doesn't
    leave higher harmonics inside the pass band untouched.

    ``envelope_method`` selects the envelope computed on the band-passed
    signal - ``"rms"`` (default) or ``"arv"``. RMS is what the literature
    specifies; ARV is kept as an explicit opt-in because it is not an RMS
    equivalent on real bursty sEMG. The choice is recorded in the returned
    ``"filter"`` mapping so a later envelope recomputation (e.g. after ECG
    gating) reuses the same method rather than silently switching.
    """
    _require_emg_recording(recording)

    metadata = dict(recording["metadata"])
    fs = float(metadata["fs"])
    array = recording["array"]
    if channel is None:
        channel = _choose_emg_channel(len(array), metadata.get("labels"))
    raw = np.asarray(array[channel], dtype=float)

    high_pass_hz: float = kwargs.pop("high_pass_hz", None) or 20.0
    low_pass_hz: float = kwargs.pop("low_pass_hz", None) or min(fs / 2 * 0.95, 500)
    compute_envelope: bool = kwargs.pop("compute_envelope", True)
    remove_ecg: bool = kwargs.pop("remove_ecg", True)
    envelope_window_seconds = kwargs.pop("envelope_window_seconds", 0.25)
    envelope_method: Literal["rms", "arv"] = kwargs.pop("envelope_method", "rms")
    notch_base_frequency: float | None = kwargs.pop("notch_base_frequency", None)
    notch_max_frequency = kwargs.pop("notch_max_frequency", fs / 2)
    notch_quality_factor = kwargs.pop(
        "notch_quality_factor", 30.0 if notch_base_frequency is not None else None
    )
    notch_before_bandpass: bool = kwargs.pop(
        "notch_before_bandpass", False if notch_base_frequency is not None else None
    )
    ecg_removal_method: Literal["gating", "wavelet"] = kwargs.pop(
        "ecg_removal_method", "gating"
    )
    fill_method = kwargs.pop("ecg_gate_fill_method", 1)

    def notch(values: np.ndarray, base_frequency: float) -> np.ndarray:
        # Default the notch's reach to Nyquist, not `low_pass_hz`: a
        # harmonic landing at or just past the low-pass cutoff (e.g. the
        # EIT frame-rate comb's 10th harmonic sitting on a 500 Hz
        # low-pass edge) is only partially attenuated by the low-pass
        # filter's finite roll-off, so it must still be fully inside the
        # notch's stopband rather than at its boundary.
        return harmonic_notch_filter(
            values,
            base_frequency=base_frequency,
            sample_frequency=fs,
            max_frequency=notch_max_frequency,
            quality_factor=notch_quality_factor,
        )

    filtered = raw
    if notch_base_frequency is not None and notch_before_bandpass:
        filtered = notch(filtered, notch_base_frequency)
    filtered = bandpass_filter(
        values=filtered,
        cutoff_frequency=(high_pass_hz, low_pass_hz),
        sample_frequency=fs,
    )
    if notch_base_frequency is not None and not notch_before_bandpass:
        filtered = notch(values=filtered, base_frequency=notch_base_frequency)

    clean = filtered
    if remove_ecg:
        ecg_peak_indexes = detect_ecg_peaks(clean, sample_frequency=fs)
        if ecg_removal_method == "gating":
            clean = gate_ecg(clean, ecg_peak_indexes, fill_method=fill_method)
        elif ecg_removal_method == "wavelet":
            clean, *_ = wavelet_denoise_ecg(
                clean, ecg_peak_indexes, sample_frequency=fs
            )
        else:
            msg = (
                f"Unknown ECG removal method '{ecg_removal_method}'; "
                "choose 'gating' or 'wavelet'."
            )
            raise ValueError(msg)
    envelope = None
    if compute_envelope:
        envelope_window_samples = max(1, int(envelope_window_seconds * fs))
        envelope = rolling_envelope(
            clean,
            window_length=envelope_window_samples,
            method=envelope_method,
        )

    return {
        **recording,
        "channel": channel,
        "fs": fs,
        "raw_channel": raw,
        "filtered": filtered,
        "clean": clean,
        "envelope": envelope,
        "filter": {
            "high_pass_hz": high_pass_hz,
            "low_pass_hz": low_pass_hz,
            "envelope_window_seconds": envelope_window_seconds,
            "envelope_method": envelope_method,
            "ecg_removal_method": ecg_removal_method,
            "ecg_gate_fill_method": fill_method,
            "notch_base_frequency": notch_base_frequency,
            "notch_max_frequency": notch_max_frequency,
            "notch_quality_factor": notch_quality_factor,
            "notch_before_bandpass": notch_before_bandpass,
        },
    }


def _detect_breaths_default(
    processed_emg: dict[str, Any],
    *,
    min_breath_width_seconds: float = 0.2,
    baseline: np.ndarray | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run ReSurfEMG EMG breath detection and return common rows."""
    if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
        msg = (
            "Default EMG breath detection expects processed EMG data from "
            "`preprocess_emg()`. Pass `detector=callable` to normalize "
            "custom detections."
        )
        raise UnsupportedWorkflowError(msg)

    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]

    if baseline is None:
        warnings.warn(
            "EMG baseline not defined; detecting breath peaks relative to "
            "zero. Run `emg.moving_baseline` or `emg.slopesum_baseline` "
            "before `emg.detect_breaths` so the detection threshold "
            "follows the drifting quiet level.",
            UserWarning,
            stacklevel=2,
        )

    min_width_samples = max(1, int(min_breath_width_seconds * fs))

    peak_indices = detect_emg_breath_peaks(
        envelope,
        baseline=baseline,
        min_peak_width_samples=min_width_samples,
        **kwargs,
    )

    events = []
    for peak_index in peak_indices:
        peak_time = int(peak_index) / fs
        events.append(
            {
                "start_time": peak_time,
                "end_time": peak_time,
                "extremum_time": peak_time,
                "start_index": int(peak_index),
                "extremum_index": int(peak_index),
                "end_index": int(peak_index),
                "sample_frequency": fs,
                "signal_name": processed_emg["channel"],
                "source": "resurfemg.detect_emg_breaths",
                "metadata": {"boundaries_measured": False},
            }
        )

    return events


def _get_enabled_functions(
    functions: dict[str, dict[str, bool]], context: _PostprocessingContext
) -> Any:
    selected_functions: dict[str, dict[str, bool]] | None = (
        dict(context.selected_functions.items()) or functions
    )
    return _normalize_selected_postprocessing(selected_functions).__contains__


def select_emg_channel(recording: Any, *, channel: int | None = None) -> dict[str, Any]:
    """Take one channel of a loaded EMG recording as the trace to analyse.

    Args:
        recording: Loaded dictionary with a channel-major array
            (n_channels, n_samples) and metadata containing fs in Hz.
        channel: Zero-based channel index. When None, channel labels select
            a breathing-muscle channel, excluding ECG/EKG labels.

    Returns:
        dict[str, Any]: The recording fields plus ``channel``, ``fs`` (Hz),
            ``raw_channel`` (the channel's samples, in the file's units)
            and an empty ``filter`` record that the filter steps fill in.

    Raises:
        TypeError: If recording lacks the required array or metadata.
        UnresolvedChannelError: If channel labels do not select an
            unambiguous EMG channel and channel is None.
    """
    _require_emg_recording(recording)
    metadata = dict(recording["metadata"])
    fs = float(metadata["fs"])
    array = recording["array"]
    if channel is None:
        channel = _choose_emg_channel(len(array), metadata.get("labels"))
    return {
        **recording,
        "channel": channel,
        "fs": fs,
        "raw_channel": np.asarray(array[channel], dtype=float),
        "filter": {},
    }


@dataclass
class _PostprocessingContext:
    envelope: np.ndarray
    baseline: np.ndarray
    peak_indices: np.ndarray
    fs: float
    baseline_window_seconds: float
    baseline_step_seconds: float
    baseline_percentile: float
    window_samples: int
    step_samples: int
    ventilator_breath_width_seconds: float
    peep: float
    ventilator_signals: dict[str, Any] | None
    ventilator_fs: float
    slope_window_seconds: float
    slope_window_samples: int
    aub_window_seconds: float
    vent_width_samples: int
    selected_functions: dict[str, dict[str, bool]]
    skipped: dict[str, str]
    computed: dict[str, Any]
    indices: dict[str, np.ndarray]


def _build_context(
    envelope: np.ndarray, fs: float, peak_indices: np.ndarray, **kwargs
) -> _PostprocessingContext:
    context = {
        "envelope": envelope,
        "fs": fs,
        "peak_indices": peak_indices,
        "baseline": None,
    }
    context["baseline_window_seconds"] = kwargs.pop("baseline_window_seconds", 7.5)
    context["baseline_step_seconds"] = kwargs.pop("baseline_step_seconds", 0.2)
    context["baseline_percentile"] = kwargs.pop("baseline_percentile", 33.0)
    context["window_samples"] = max(
        1, int(context["baseline_window_seconds"] * context["fs"])
    )
    context["step_samples"] = max(
        1, int(context["baseline_step_seconds"] * context["fs"])
    )
    context["ventilator_breath_width_seconds"] = kwargs.pop(
        "ventilator_breath_width_seconds", 0.1
    )

    context["ventilator_signals"] = ventilator_signals(
        ventilator=kwargs.pop("ventilator", None),
        airway_pressure_channel=kwargs.pop("ventilator_pressure_channel", 0),
        flow_channel=kwargs.pop("ventilator_flow_channel", 1),
        volume_channel=kwargs.pop("ventilator_volume_channel", 2),
    )
    context["ventilator_fs"] = context["ventilator_signals"]["fs"] or None
    context["peep"] = kwargs.pop(
        "peep",
        float(np.nanmedian(context["ventilator_signals"]["pressure"]))
        if context["ventilator_signals"] is not None
        else 0.0,
    )
    context["vent_width_samples"] = max(
        1, int(context["ventilator_breath_width_seconds"] * context["ventilator_fs"])
    )

    context["slope_window_seconds"] = kwargs.pop("slope_window_seconds", 0.5)
    context["slope_window_samples"] = max(
        1, int(context["slope_window_seconds"] * context["fs"])
    )
    context["aub_window_seconds"] = kwargs.pop("aub_window_seconds", 5.0)
    context["selected_functions"] = kwargs.pop(
        "selected_functions",
        {
            **BASELINE_FUNCTIONS,
            **EVENT_DETECTION_FUNCTIONS,
            **VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
            **FEATURES_FUNCTIONS,
            **EMG_QUALITY_ASSESSMENT_FUNCTIONS,
            **MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS,
        },
    )
    context["computed"] = {
        "baseline": {},
        "event_detection": {},
        "features": {},
        "quality_assessment": {},
    }
    context["skipped"] = {}
    context["indices"] = {
        "start_indices": np.asarray([], dtype=int),
        "end_indices": np.asarray([], dtype=int),
        "start_end_validity": np.asarray([], dtype=bool),
    }

    return _PostprocessingContext(**context)


def _compute_baseline_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:

    enabled = _get_enabled_functions(BASELINE_FUNCTIONS, context)

    if enabled(("baseline", "moving_baseline")):
        moving_baseline = compute_moving_baseline(
            context.envelope,
            window_samples=context.window_samples,
            step_samples=context.step_samples,
            percentile=context.baseline_percentile,
        )
        context.computed["baseline"]["moving_baseline"] = context.baseline = (
            moving_baseline
        )

    if enabled(("baseline", "slopesum_baseline")):
        slopesum_baseline = compute_slopesum_baseline(
            context.envelope,
            window_samples=context.window_samples,
            step_samples=context.step_samples,
            sample_frequency=context.fs,
            percentile=context.baseline_percentile,
            moving_average_samples=max(1, int(context.fs // 2)),
            percentile_window_samples=max(1, int(context.fs)),
        )
        context.baseline = slopesum_baseline[0]
        context.computed["baseline"]["slopesum_baseline"] = {
            "baseline": slopesum_baseline[0],
            "running_mean": slopesum_baseline[1],
            "running_std": slopesum_baseline[2],
            "series": slopesum_baseline[3],
        }
    return context


def _ventilator_breaths_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:

    enabled = _get_enabled_functions(
        functions={
            **EVENT_DETECTION_FUNCTIONS,
            **VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
        },
        context=context,
    )
    if context.ventilator_signals is None:
        context.skipped.update(
            {
                "event_detection.find_occluded_breaths": "Needs ventilator pressure and PEEP inputs.",
                "event_detection.detect_ventilator_breath": "Needs ventilator volume input.",
                "quality_assessment.detect_non_consecutive_manoeuvres": "Needs ventilator breath and manoeuvre indices.",
                "quality_assessment.ventilator_respiratory_rate": "Needs ventilator volume input.",
            }
        )
        return context
    ventilator_breath_indices = np.asarray([], dtype=int)
    if enabled(("event_detection", "detect_ventilator_breath")):
        ventilator_breath_indices = np.asarray(
            a=detect_ventilator_breath_peaks(
                volume=context.ventilator_signals["volume"],
                sample_frequency=context.ventilator_fs,
                start_index=0,
                end_index=len(context.ventilator_signals["volume"]) - 1,
                width_samples=context.vent_width_samples,
            ),
            dtype=int,
        )
        context.computed["event_detection"]["detect_ventilator_breath"] = (
            ventilator_breath_indices
        )

    pocc_indices = np.asarray([], dtype=int)
    if enabled(("event_detection", "find_occluded_breaths")):
        pocc_indices = np.asarray(
            a=detect_occluded_breath_peaks(
                pressure=context.ventilator_signals["pressure"],
                sample_frequency=context.ventilator_fs,
                peep=context.peep,
            ),
            dtype=int,
        )
        context.computed["event_detection"]["find_occluded_breaths"] = pocc_indices

    if enabled(("quality_assessment", "detect_non_consecutive_manoeuvres")):
        if len(ventilator_breath_indices) and len(pocc_indices):
            context.computed["quality_assessment"][
                "detect_non_consecutive_manoeuvres"
            ] = detect_non_consecutive_manoeuvres(
                ventilator_breath_indices,
                pocc_indices,
            )
        else:
            context.skipped["quality_assessment.detect_non_consecutive_manoeuvres"] = (
                "Needs ventilator breath and manoeuvre indices."
            )

    if (
        enabled(("quality_assessment", "evaluate_respiratory_rates"))
        and len(ventilator_breath_indices) >= 2
    ):
        context.computed["quality_assessment"]["ventilator_respiratory_rate"] = (
            respiratory_rate_from_indices(
                indices=ventilator_breath_indices,
                sample_frequency=context.ventilator_fs,
            )
        )
    else:
        context.skipped["quality_assessment.ventilator_respiratory_rate"] = (
            "Needs at least two ventilator breaths."
        )
    return context


def _check_baseline_and_peaks(
    context: _PostprocessingContext,
    selected_functions: dict[str, dict[str, bool]],
    enabled: Callable[[tuple[str, str]], bool],
) -> tuple[_PostprocessingContext, int]:
    skipped: int = 0
    if len(context.peak_indices) < 2 or context.baseline is None:
        for category, functions in selected_functions.items():
            for name in functions:
                if enabled:
                    context.skipped[f"{category}.{name}"] = (
                        "Needs detected EMG breath peaks and baseline."
                    )
                    skipped += 1
    return context, skipped


def _detect_emg_events_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:
    enabled = _get_enabled_functions(
        functions=EVENT_DETECTION_FUNCTIONS, context=context
    )
    context, skipped_update = _check_baseline_and_peaks(
        context, selected_functions=EVENT_DETECTION_FUNCTIONS, enabled=enabled
    )
    if skipped_update:
        return context
    if enabled(("event_detection", "onoffpeak_baseline_crossing")):
        context.computed["event_detection"]["onoffpeak_baseline_crossing"] = (
            onoff_from_baseline_crossings(
                signal=context.envelope,
                baseline=context.baseline,
                peak_indices=context.peak_indices,
            )
        )
        start_indices, end_indices, _valid_starts, _valid_ends, valid_peaks = (
            context.computed["event_detection"]["onoffpeak_baseline_crossing"]
        )
        context.indices["start_indices"] = start_indices
        context.indices["end_indices"] = end_indices
        context.indices["start_end_validity"] = np.asarray(valid_peaks, dtype=bool)
    slope_window_samples = max(1, int(context.slope_window_seconds * context.fs))
    if enabled(("event_detection", "onoffpeak_slope_extrapolation")):
        context.computed["event_detection"]["onoffpeak_slope_extrapolation"] = (
            onoff_from_slope(
                signal=context.envelope,
                sample_frequency=context.fs,
                peak_indices=context.peak_indices,
                slope_window=slope_window_samples,
            )
        )
        start_indices, end_indices, _valid_starts, _valid_ends, valid_peaks = (
            context.computed["event_detection"]["onoffpeak_slope_extrapolation"]
        )
        context.indices["start_indices"] = start_indices
        context.indices["end_indices"] = end_indices
        context.indices["start_end_validity"] = np.asarray(valid_peaks, dtype=bool)
    return context


def _compute_emg_features_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:
    enabled = _get_enabled_functions(FEATURES_FUNCTIONS, context)
    context, skipped_updated = _check_baseline_and_peaks(
        context, FEATURES_FUNCTIONS, enabled
    )
    if (
        context.indices["start_indices"] is None
        or context.indices["end_indices"] is None
    ):
        context.skipped["features"] += (
            "Needs detected EMG breath start and end indices."
        )
        return context
    if skipped_updated:
        return context

    if enabled(("features", "time_to_peak")):
        context.computed["features"]["time_to_peak"] = time_to_peak(
            values=context.envelope,
            start_indices=context.indices["start_indices"],
            end_indices=context.indices["end_indices"],
        )
    if enabled(("features", "pseudo_slope")):
        context.computed["features"]["pseudo_slope"] = pseudo_slope(
            values=context.envelope,
            start_indices=context.indices["start_indices"],
            end_indices=context.indices["end_indices"],
        )
    if enabled(("features", "amplitude")):
        context.computed["features"]["amplitude"] = amplitude_at_peaks(
            values=context.envelope,
            peak_indices=context.peak_indices,
            baseline=context.baseline,
        )
    if enabled(("features", "time_product")):
        context.computed["features"]["time_product"] = window_integral(
            values=context.envelope,
            sample_frequency=context.fs,
            start_indices=context.indices["start_indices"],
            end_indices=context.indices["end_indices"],
            baseline=context.baseline,
        )
    if enabled(("features", "area_under_baseline")):
        context.computed["features"]["area_under_baseline"] = area_under_baseline(
            values=context.envelope,
            sample_frequency=context.fs,
            peak_indices=context.peak_indices,
            start_indices=context.indices["start_indices"],
            end_indices=context.indices["end_indices"],
            window=max(1, int(context.aub_window_seconds * context.fs)),
            baseline=context.baseline,
        )
    if enabled(("features", "respiratory_rate")) and len(context.peak_indices) >= 2:
        _, context.computed["features"]["respiratory_rate"] = (
            respiratory_rate_from_indices(
                indices=context.peak_indices, sample_frequency=context.fs
            )
        )

    elif enabled(("features", "respiratory_rate")):
        context.skipped["features"] += "Needs at least two EMG breaths."

    return context


def _compute_emg_quality_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:

    enabled = _get_enabled_functions(
        functions=EMG_QUALITY_ASSESSMENT_FUNCTIONS, context=context
    )
    context, skipped_updated = _check_baseline_and_peaks(
        context=context,
        selected_functions=EMG_QUALITY_ASSESSMENT_FUNCTIONS,
        enabled=enabled,
    )
    if not skipped_updated:
        return context
    if enabled(("quality_assessment", "snr_pseudo")):
        context.computed["quality_assessment"]["snr_pseudo"] = snr_pseudo(
            envelope=context.envelope,
            peak_indices=context.peak_indices,
            baseline=context.baseline,
            sample_frequency=context.fs,
        )
    if (
        enabled(("quality_assessment", "percentage_under_baseline"))
        and context.indices["start_indices"] is not None
        and context.indices["end_indices"] is not None
    ):
        context.computed["quality_assessment"]["percentage_under_baseline"] = (
            percentage_under_baseline(
                signal=context.envelope,
                peak_indices=context.peak_indices,
                start_indices=context.indices["start_indices"],
                end_indices=context.indices["end_indices"],
                baseline=context.baseline,
                sample_frequency=context.fs,
            )
        )
    if (
        enabled(("quality_assessment", "detect_local_high_aub"))
        and "area_under_baseline" in context.computed["features"]
    ):
        aubs = context.computed["features"]["area_under_baseline"][0]
        context.computed["quality_assessment"]["detect_local_high_aub"] = (
            detect_local_high_aub(aubs)
        )
    if (
        enabled(("quality_assessment", "detect_extreme_time_products"))
        and "time_product" in context.computed["features"]
    ):
        time_products = context.computed["features"]["time_product"]
        context.computed["quality_assessment"]["detect_extreme_time_products"] = (
            detect_extreme_time_products(time_products)
        )
    if (
        enabled(("quality_assessment", "evaluate_bell_curve_error"))
        and context.indices["start_indices"] is not None
        and context.indices["end_indices"] is not None
        and "time_product" in context.computed["features"]
    ):
        time_products = context.computed["features"]["time_product"]
        context.computed["quality_assessment"]["evaluate_bell_curve_error"] = (
            evaluate_bell_curve_error(
                peak_indices=context.peak_indices,
                start_indices=context.indices["start_indices"],
                end_indices=context.indices["end_indices"],
                signal=context.envelope,
                time_products=time_products,
                sample_frequency=context.fs,
            )
        )
    return context


def _compute_multimodal_quality_default(
    context: _PostprocessingContext,
) -> _PostprocessingContext:
    enabled = _get_enabled_functions(
        functions=MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS, context=context
    )
    skipped: dict[str, str] = {}

    context, skipped_updated = _check_baseline_and_peaks(
        context=context,
        selected_functions=MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS,
        enabled=enabled,
    )
    if not skipped_updated:
        return context
    ventilator_breath_indices = context.computed.get("event_detection", {}).get(
        "detect_ventilator_breath", np.asarray(a=[], dtype=int)
    )
    if (
        enabled(("quality_assessment", "evaluate_event_timing"))
        and context.ventilator_signals is not None
        and len(ventilator_breath_indices)
    ):
        peak_indices = context.computed.get("event_detection", {}).get(
            "detect_peaks", np.asarray(a=[], dtype=int)
        )
        paired_count = min(len(peak_indices), len(ventilator_breath_indices))
        context.computed["quality_assessment"]["evaluate_event_timing"] = (
            evaluate_event_timing(
                peak_indices[:paired_count] / context.fs,
                ventilator_breath_indices[:paired_count] / float(context.ventilator_fs),
            )
        )
    elif enabled(("quality_assessment", "evaluate_event_timing")):
        skipped["quality_assessment.evaluate_event_timing"] = (
            "Needs ventilator breath timing."
        )

    # Deliberately independent of 'evaluate_event_timing' above: its
    # only real prerequisite is that 'ventilator_respiratory_rate' was
    # computed (near the top of this function), which does not
    # require 'evaluate_event_timing' to be selected. Nesting this
    # under that block previously meant selecting
    # 'evaluate_respiratory_rates' alone (without also selecting
    # 'evaluate_event_timing') silently produced nothing - no result,
    # no skip reason.
    if (
        enabled(("quality_assessment", "evaluate_respiratory_rates"))
        and "ventilator_respiratory_rate" in context.computed["quality_assessment"]
    ):
        rr_vent = context.computed["quality_assessment"]["ventilator_respiratory_rate"][
            0
        ]
        context.computed["quality_assessment"]["evaluate_respiratory_rates"] = (
            evaluate_respiratory_rates(
                context.peak_indices,
                len(context.envelope) / context.fs,
                rr_vent,
            )
        )
    elif enabled(("quality_assessment", "evaluate_respiratory_rates")):
        skipped["quality_assessment.evaluate_respiratory_rates"] = (
            "Needs ventilator respiratory rate (requires "
            "'evaluate_respiratory_rates' selected together with at "
            "least two ventilator breaths)."
        )

    return context


# TODO
def _postprocess_default(
    processed_emg: Any,
    events: Sequence[BreathEvent] | None = None,
    **kwargs,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        msg = "EMG postprocessing requires numpy."
        raise OptionalDependencyError(msg) from exc

    if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
        msg = (
            "Default EMG postprocessing expects processed EMG data from "
            "`preprocess_emg()`."
        )
        raise UnsupportedWorkflowError(msg)
    context: _PostprocessingContext = _build_context(
        envelope=processed_emg["envelope"],
        fs=processed_emg["fs"],
        peak_indices=peak_indices_from_events(events, processed_emg["fs"]),
        **kwargs,
    )

    selected = _normalize_selected_postprocessing(context.selected_functions)
    enabled = selected.__contains__

    # since EMG methods are now native, there's no need for the ReSurfEMG dependency check
    context = _compute_baseline_default(context)

    context = _ventilator_breaths_default(context)

    context = _detect_emg_events_default(context)

    context = _compute_emg_features_default(context)

    context = _compute_emg_quality_default(context)

    context = _compute_multimodal_quality_default(context)
    if context.baseline is None or len(context.peak_indices) < 2:
        if enabled(("quality_assessment", "evaluate_event_timing")):
            context.skipped["quality_assessment.evaluate_event_timing"] = (
                "Needs detected EMG breath peaks."
            )
        if enabled(("quality_assessment", "evaluate_respiratory_rates")):
            context.skipped["quality_assessment.evaluate_respiratory_rates"] = (
                "Needs detected EMG breath peaks."
            )
        if enabled(("quality_assessment", "pocc_quality")):
            context.skipped["quality_assessment.pocc_quality"] = (
                "Needs Pocc ventilator inputs."
            )
        if enabled(("quality_assessment", "interpeak_dist")):
            context.skipped["quality_assessment.interpeak_dist"] = (
                "Needs ECG peak indices."
            )
    # TODO update settings
    return {
        # "available": available_postprocessing(),
        "computed": context.computed,
        "skipped": context.skipped,
        "peak_indices": context.peak_indices,
        "settings": {
            "baseline_window_seconds": context.baseline_window_seconds,
            "baseline_step_seconds": context.baseline_step_seconds,
            "baseline_percentile": context.baseline_percentile,
            "slope_window_seconds": context.slope_window_seconds,
            "aub_window_seconds": context.aub_window_seconds,
            "ventilator_breath_width_seconds": context.ventilator_breath_width_seconds,
            "peep": context.peep,
        },
    }


class _DefaultsMixin:
    _preprocess_default = staticmethod(_preprocess_default)
    _postprocess_default = staticmethod(_postprocess_default)
    _detect_breaths_default = staticmethod(_detect_breaths_default)
    default_low_pass_hz = staticmethod(default_low_pass_hz)
    notch_max_frequency_hz = staticmethod(notch_max_frequency_hz)
    select_emg_channel = staticmethod(select_emg_channel)
