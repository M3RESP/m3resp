"""Default EMG preprocessing and postprocessing functions for the ReSurfEMG workflow.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""  # noqa: CPY001

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import m3resp.processing.intervals
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.processing.filters import bandpass_filter, harmonic_notch_filter
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
from m3resp.processing.ventilator import estimate_peep
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
    interpeak_distance,
    percentage_under_baseline,
    snr_pseudo,
)
from .shared import (
    BASELINE_FUNCTIONS,
    EMG_EVENT_DETECTION_FUNCTIONS,
    EMG_PROCESSING_DEFAULT_PARAMETERS,
    EMG_QUALITY_ASSESSMENT_FUNCTIONS,
    FEATURES_FUNCTIONS,
    MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS,
    VENTILATOR_EVENT_DETECTION_FUNCTIONS,
    VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
    _choose_ecg_channel,
    _choose_emg_channel,
    _normalize_selected_postprocessing,
    _require_emg_recording,
)
from .signals import peak_indices_from_events, ventilator_signals

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from m3resp.data.events import BreathEvent


def _merge_selections(
    *selections: dict[str, dict[str, bool]],
) -> dict[str, dict[str, bool]]:
    merged: dict[str, dict[str, bool]] = {}
    for selection in selections:
        for category, functions in selection.items():
            merged.setdefault(category, {}).update(functions)
    return merged


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


def emg_envelope(
    values: Mapping,
    *,
    sample_frequency: float,
    window_seconds: float = 0.25,
    method: Literal["rms", "arv", "median"] = "rms",
) -> np.ndarray:
    """Compute the envelope of an EMG trace over a moving window.

    RMS is the envelope used in the literature. ARV gives different values
    from RMS on real sEMG. The median envelope (median of the absolute
    signal) ignores short spikes such as heartbeat leftovers.

    Args:
        values: EMG samples, one value per sample.
        sample_frequency: Sampling rate in Hz.
        window_seconds: Window length in seconds, converted to at least
            one sample (rounded down).
        method: ``"rms"``, ``"arv"`` or ``"median"``.

    Returns:
        numpy.ndarray: The envelope, one value per sample, in the units of
            the trace.
    """
    return rolling_envelope(
        values,
        window_length=max(1, int(window_seconds * sample_frequency)),
        method=method,
    )


def _preprocess_default(
    recording: Mapping,
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
    channel_labels = metadata.get("labels")
    if channel is None:
        channel = _choose_emg_channel(len(array), channel_labels)
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
        ecg_channel = kwargs.pop(
            "ecg_channel",
            _choose_ecg_channel(channel_labels) if channel_labels else None,
        )
        ecg_signal = (
            np.asarray(array[ecg_channel[0]], dtype=float)
            if ecg_channel is not None
            else clean
        )
        ecg_peak_indexes = detect_ecg_peaks(ecg_signal, sample_frequency=fs)
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
    processed_emg: Mapping,
    *,
    min_breath_width_seconds: float = 0.2,
    baseline: np.ndarray | None = None,
    **kwargs,
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
    functions: dict[str, dict[str, bool]],  # noqa: ARG001
    context: _PostprocessingContext,
) -> Callable[[tuple[str, str]], bool]:
    return lambda key: _is_selected(context, *key)


def select_emg_channel(
    recording: Mapping, *, channel: int | None = None
) -> dict[str, Any]:
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
    ecg_peak_indices: np.ndarray
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


def _is_selected(context: _PostprocessingContext, category: str, function: str) -> bool:
    return bool(context.selected_functions.get(category, {}).get(function, False))


def _disable_selected_function(
    context: _PostprocessingContext, category: str, function: str, reason: str
) -> None:
    """Disable a selected postprocessing function, recording the reason."""
    if _is_selected(context, category, function):
        context.selected_functions[category][function] = False
        context.skipped[f"{category}.{function}"] = reason


def _build_context(
    envelope: np.ndarray, fs: float, peak_indices: np.ndarray, **kwargs
) -> _PostprocessingContext:
    context = {
        "envelope": envelope,
        "fs": fs,
        "peak_indices": peak_indices,
        "baseline": None,
    }
    context["ecg_peak_indices"] = kwargs.pop("ecg_peak_indices", None)
    for param, default in EMG_PROCESSING_DEFAULT_PARAMETERS.items():
        context[f"{param}"] = kwargs.pop(f"{param}", default)
    context["window_samples"] = max(
        1, int(context["baseline_window_seconds"] * context["fs"])
    )
    context["step_samples"] = max(
        1, int(context["baseline_step_seconds"] * context["fs"])
    )
    context["ventilator_signals"] = ventilator_signals(
        ventilator=kwargs.pop("ventilator", None),
        airway_pressure_channel=kwargs.pop("ventilator_pressure_channel", None),
        flow_channel=kwargs.pop("ventilator_flow_channel", None),
        volume_channel=kwargs.pop("ventilator_volume_channel", None),
    )
    context["ventilator_fs"] = (
        context["ventilator_signals"]["fs"]
        if context["ventilator_signals"] is not None
        else None
    )
    context["peep"] = kwargs.pop(
        "peep",
        estimate_peep(
            context["ventilator_signals"]["airway_pressure"],
            context["ventilator_signals"]["volume"],
        )
        if context["ventilator_signals"] is not None
        else float("nan"),
    )
    context["vent_width_samples"] = (
        max(
            1,
            int(context["ventilator_breath_width_seconds"] * context["ventilator_fs"]),
        )
        if context["ventilator_fs"] is not None
        else None
    )
    context["slope_window_samples"] = max(
        1, int(context["slope_window_seconds"] * context["fs"])
    )

    context["selected_functions"] = copy.deepcopy(
        kwargs.pop(
            "selected_functions",
            _merge_selections(
                BASELINE_FUNCTIONS,
                EMG_EVENT_DETECTION_FUNCTIONS,
                VENTILATOR_EVENT_DETECTION_FUNCTIONS,
                VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
                FEATURES_FUNCTIONS,
                EMG_QUALITY_ASSESSMENT_FUNCTIONS,
                MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS,
            ),
        )
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
) -> None:

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


def _ventilator_breaths_default(
    context: _PostprocessingContext,
) -> None:

    enabled = _get_enabled_functions(
        functions=_merge_selections(
            VENTILATOR_EVENT_DETECTION_FUNCTIONS,
            VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
        ),
        context=context,
    )
    if context.ventilator_signals is None:
        return
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
                pressure=context.ventilator_signals["airway_pressure"],
                sample_frequency=context.ventilator_fs,
                peep=context.peep,
            ),
            dtype=int,
        )
        context.computed["event_detection"]["find_occluded_breaths"] = pocc_indices

    if len(ventilator_breath_indices) < 2:
        _disable_selected_function(
            context,
            "quality_assessment",
            "ventilator_respiratory_rate",
            "Needs at least two ventilator breaths.",
        )
    if len(ventilator_breath_indices) == 0 or len(pocc_indices) == 0:
        _disable_selected_function(
            context,
            "quality_assessment",
            "detect_non_consecutive_manoeuvres",
            "Needs ventilator breath and manoeuvre indices.",
        )
    if enabled(("quality_assessment", "ventilator_respiratory_rate")):
        context.computed["quality_assessment"]["ventilator_respiratory_rate"] = (
            respiratory_rate_from_indices(
                indices=ventilator_breath_indices,
                sample_frequency=context.ventilator_fs,
            )
        )

    if enabled(("quality_assessment", "detect_non_consecutive_manoeuvres")):
        context.computed["quality_assessment"]["detect_non_consecutive_manoeuvres"] = (
            detect_non_consecutive_manoeuvres(
                ventilator_breath_indices,
                pocc_indices,
            )
        )


def _detect_emg_events_default(
    context: _PostprocessingContext,
) -> None:
    enabled = _get_enabled_functions(
        functions=EMG_EVENT_DETECTION_FUNCTIONS, context=context
    )

    if enabled(("event_detection", "onoffpeak_baseline_crossing")):
        context.computed["event_detection"]["onoffpeak_baseline_crossing"] = (
            m3resp.processing.intervals.onoff_from_baseline_crossings(
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
            m3resp.processing.intervals.onoff_from_slope(
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


_BOUNDS_REASON = "Needs detected EMG breath start and end indices."


def _compute_emg_features_default(
    context: _PostprocessingContext,
) -> None:
    enabled = _get_enabled_functions(FEATURES_FUNCTIONS, context)

    starts = context.indices["start_indices"]
    ends = context.indices["end_indices"]
    has_bounds = len(starts) > 0 and len(ends) > 0

    if enabled(("features", "amplitude")):
        context.computed["features"]["amplitude"] = amplitude_at_peaks(
            values=context.envelope,
            peak_indices=context.peak_indices,
            baseline=context.baseline,
        )

    if enabled(("features", "respiratory_rate")):
        _, context.computed["features"]["respiratory_rate"] = (
            respiratory_rate_from_indices(
                indices=context.peak_indices, sample_frequency=context.fs
            )
        )
    if not has_bounds:
        for name in (
            "time_to_peak",
            "pseudo_slope",
            "time_product",
            "area_under_baseline",
        ):
            _disable_selected_function(
                context,
                "features",
                name,
                _BOUNDS_REASON,
            )
        return

    if enabled(("features", "time_to_peak")):
        context.computed["features"]["time_to_peak"] = time_to_peak(
            values=context.envelope,
            start_indices=starts,
            end_indices=ends,
        )
    if enabled(("features", "pseudo_slope")):
        context.computed["features"]["pseudo_slope"] = pseudo_slope(
            values=context.envelope,
            start_indices=starts,
            end_indices=ends,
        )
    if enabled(("features", "time_product")):
        context.computed["features"]["time_product"] = window_integral(
            values=context.envelope,
            sample_frequency=context.fs,
            start_indices=starts,
            end_indices=ends,
            baseline=context.baseline,
        )
    if enabled(("features", "area_under_baseline")):
        context.computed["features"]["area_under_baseline"] = area_under_baseline(
            values=context.envelope,
            sample_frequency=context.fs,
            peak_indices=context.peak_indices,
            start_indices=starts,
            end_indices=ends,
            window=max(1, int(context.aub_window_seconds * context.fs)),
            baseline=context.baseline,
        )


def _compute_emg_quality_default(
    context: _PostprocessingContext,
) -> None:

    enabled = _get_enabled_functions(
        functions=EMG_QUALITY_ASSESSMENT_FUNCTIONS, context=context
    )

    starts = context.indices["start_indices"]
    ends = context.indices["end_indices"]
    is_bounded = len(starts) > 0 and len(ends) > 0

    features = context.computed["features"]
    if not is_bounded:
        for name in ("percentage_under_baseline", "evaluate_bell_curve_error"):
            _disable_selected_function(
                context, "quality_assessment", name, _BOUNDS_REASON
            )
    if "time_product" not in features:
        for name in ("detect_extreme_time_products", "evaluate_bell_curve_error"):
            _disable_selected_function(
                context,
                "quality_assessment",
                name,
                "Needs computed EMG breath time products.",
            )
    if "area_under_baseline" not in features:
        _disable_selected_function(
            context,
            "quality_assessment",
            "detect_local_high_aub",
            "Needs computed EMG breath area-under-baseline.",
        )
    if enabled(("quality_assessment", "snr_pseudo")):
        context.computed["quality_assessment"]["snr_pseudo"] = snr_pseudo(
            envelope=context.envelope,
            peak_indices=context.peak_indices,
            baseline=context.baseline,
            sample_frequency=context.fs,
        )
    if enabled(("quality_assessment", "percentage_under_baseline")):
        context.computed["quality_assessment"]["percentage_under_baseline"] = (
            percentage_under_baseline(
                signal=context.envelope,
                peak_indices=context.peak_indices,
                start_indices=starts,
                end_indices=ends,
                baseline=context.baseline,
                sample_frequency=context.fs,
            )
        )
    if enabled(("quality_assessment", "detect_local_high_aub")):
        aubs = context.computed["features"]["area_under_baseline"][0]
        context.computed["quality_assessment"]["detect_local_high_aub"] = (
            detect_local_high_aub(aubs)
        )
    if enabled(("quality_assessment", "detect_extreme_time_products")):
        time_products = context.computed["features"]["time_product"]
        context.computed["quality_assessment"]["detect_extreme_time_products"] = (
            detect_extreme_time_products(time_products)
        )
    if enabled(("quality_assessment", "evaluate_bell_curve_error")):
        time_products = context.computed["features"]["time_product"]
        context.computed["quality_assessment"]["evaluate_bell_curve_error"] = (
            evaluate_bell_curve_error(
                peak_indices=context.peak_indices,
                start_indices=starts,
                end_indices=ends,
                signal=context.envelope,
                time_products=time_products,
                sample_frequency=context.fs,
            )
        )
    if enabled(("quality_assessment", "interpeak_distance")):
        context.computed["quality_assessment"]["interpeak_distance"] = (
            interpeak_distance(
                ecg_peak_indices=context.ecg_peak_indices,
                emg_peak_indices=context.peak_indices,
            )
        )


def _compute_multimodal_quality_default(
    context: _PostprocessingContext,
) -> None:
    enabled = _get_enabled_functions(
        functions=MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS, context=context
    )
    ventilator_breath_indices = context.computed.get("event_detection", {}).get(
        "detect_ventilator_breath", np.asarray(a=[], dtype=int)
    )
    if not len(ventilator_breath_indices):
        _disable_selected_function(
            context,
            "quality_assessment",
            "evaluate_event_timing",
            "Needs ventilator breath timing.",
        )
    if enabled(("quality_assessment", "evaluate_event_timing")):
        peak_indices = context.peak_indices
        paired_count = min(len(peak_indices), len(ventilator_breath_indices))
        context.computed["quality_assessment"]["evaluate_event_timing"] = (
            evaluate_event_timing(
                peak_indices[:paired_count] / context.fs,
                ventilator_breath_indices[:paired_count] / float(context.ventilator_fs),
            )
        )
    if "ventilator_respiratory_rate" not in context.computed["quality_assessment"]:
        _disable_selected_function(
            context,
            "quality_assessment",
            "evaluate_respiratory_rates",
            "Needs computed ventilator respiratory rate.",
        )
    if enabled(("quality_assessment", "evaluate_respiratory_rates")):
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


def _postprocess_default(
    processed_emg: Mapping,
    events: Sequence[BreathEvent] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run the default EMG postprocessing pipeline and return a dictionary of results.

    Args:
        processed_emg: the processed EMG bundle returned by `_preprocess_default`;
            must contain "envelope" and "fs"
        events (Sequence[BreathEvent], Optional): detected EMG breaths, e.g. from
            `_detect_breaths_default`. Their extremum indices are the breath
            peaks every peak-based step uses. If None or fewer than two, the
            peak-based steps are skipped.
        **kwargs: optional keyword arguments for postprocessing customization

    Returns:
        dict[str, Any]: a dictionary containing the postprocessing results
            - "computed": the results, grouped by category ("baseline",
                "event_detection", "features", "quality_assessment") and keyed
                by function name
            - "skipped": the selected functions that did not run, keyed by
                "category.function", with the reason as value
            - "peak_indices": the EMG breath peak indices (samples) taken from
                ``events``
            - "settings": the window, percentile and PEEP settings used

    Raises:
        UnsupportedWorkflowError: if ``processed_emg`` is not a dictionary with
            an "envelope" entry.

    Without any additional arguments, the function runs every function listed
    below on the EMG envelope. Functions whose inputs are missing (e.g. no
    ventilator recording, fewer than two breaths, no ECG peaks) are skipped
    and listed in "skipped" with the reason, instead of raising an error.
    Every selected function therefore appears in exactly one of "computed"
    or "skipped".

    The pipeline runs in this order:
    -- baseline computation
    -- ventilator breath and occlusion detection (if a ventilator is given)
    -- EMG breath onset and offset detection
    -- EMG breath features
    -- EMG quality assessment
    -- multimodal (EMG + ventilator) quality assessment

    # Postprocessing pipeline customization
    ## Function selection
    - ``selected_functions``: nested mapping ``{category: {function: bool}}``
        of the functions to run (default: all functions listed below). The
        mapping is copied, so the caller's dictionary is never modified.

    ## Baseline
    Functions (category "baseline"):
    - ``moving_baseline``: running percentile of the envelope
    - ``slopesum_baseline``: slope-sum baseline; also returns its running
        mean, running standard deviation and slope series
    When both are selected, the slope-sum baseline is the one used by all
    later steps.

    Parameters:
    - ``baseline_window_seconds``: window length in seconds (default: 7.5)
    - ``baseline_step_seconds``: step between windows in seconds (default: 0.2)
    - ``baseline_percentile``: percentile of the envelope used as baseline
        (default: 33.0)

    ## Event detection
    EMG functions (category "event_detection"), which need at least two breath
    peaks and a baseline:
    - ``onoffpeak_baseline_crossing``: breath onsets and offsets where the
        envelope crosses the baseline
    - ``onoffpeak_slope_extrapolation``: breath onsets and offsets
        extrapolated from the slope around each peak
    When both are selected, the slope-extrapolated onsets and offsets are the
    ones used by the features and quality steps.

    Ventilator functions (category "event_detection"):
    - ``detect_ventilator_breath``: breath peaks in the ventilator volume
    - ``find_occluded_breaths``: occlusion manoeuvres in the airway pressure;
        needs ``peep``

    Parameters:
    - ``slope_window_seconds``: window for slope extrapolation in seconds
        (default: 0.5)
    - ``ventilator_breath_width_seconds``: minimum ventilator breath width in
        seconds (default: 0.1)

    ## Ventilator input
    - ``ventilator``: the loaded ventilator recording (default: None). Without
        it, all ventilator and multimodal functions are skipped.
    - ``ventilator_pressure_channel``, ``ventilator_flow_channel``,
        ``ventilator_volume_channel``: channels to use from the ventilator
        recording (default: None, chosen automatically)
    - ``peep``: positive end-expiratory pressure, in the pressure channel's
        units (default: NaN). Without it, ``find_occluded_breaths`` is skipped,
        because PEEP cannot be estimated reliably from the pressure trace.

    ## Features
    Functions (category "features"):
    - ``amplitude``: envelope amplitude above baseline at each breath peak
    - ``respiratory_rate``: per-breath respiratory rate from the peak
        intervals; needs only two or more peaks
    - ``time_to_peak``: time from breath onset to peak
    - ``pseudo_slope``: rise of the envelope from onset to peak
    - ``time_product``: envelope integral above baseline between onset and
        offset
    - ``area_under_baseline``: area of the envelope below the baseline
        around each breath
    ``time_to_peak``, ``pseudo_slope``, ``time_product`` and
    ``area_under_baseline`` also need breath onsets and offsets.

    Parameters:
    - ``aub_window_seconds``: window for the area under the baseline in
        seconds (default: 5.0)

    ## Quality assessment
    EMG functions (category "quality_assessment"):
    - ``snr_pseudo``: signal-to-noise ratio of each breath against the baseline
    - ``percentage_under_baseline``: share of each breath below the baseline;
        needs onsets and offsets
    - ``detect_local_high_aub``: breaths with an unusually high area under the
        baseline; needs ``area_under_baseline``
    - ``detect_extreme_time_products``: outlier time products; needs
        ``time_product``
    - ``evaluate_bell_curve_error``: deviation of each breath from a bell
        shape; needs onsets, offsets and ``time_product``
    - ``interpeak_distance``: distance between EMG breath peaks and ECG peaks;
        needs ``ecg_peak_indices``

    Ventilator functions (category "quality_assessment"):
    - ``ventilator_respiratory_rate``: respiratory rate from the ventilator
        breaths; needs at least two ventilator breaths
    - ``detect_non_consecutive_manoeuvres``: occlusion manoeuvres not
        separated by normal breaths; needs ventilator breaths and occlusions

    Multimodal functions (category "quality_assessment"):
    - ``evaluate_event_timing``: timing difference between EMG and ventilator
        breaths, paired in order of occurrence
    - ``evaluate_respiratory_rates``: agreement between the EMG and
        ventilator respiratory rates; needs ``ventilator_respiratory_rate``

    Parameters:
    - ``ecg_peak_indices``: ECG R-peak indices in envelope samples, e.g. from
        ECG removal during preprocessing (default: None). At least two are
        needed for ``interpeak_distance``.

    # Additional information
    All indices are in samples of the envelope, except the ventilator
    indices, which are in samples of the ventilator recording. Durations are
    converted to samples with the respective sampling rate.

    ``evaluate_event_timing`` pairs the first EMG breath with the first
    ventilator breath, the second with the second, and so on. It assumes both
    recordings are synchronized and that neither modality has missed or extra
    breaths; check ``evaluate_respiratory_rates`` before relying on it.

    The default parameter values are defined in
    ``EMG_PROCESSING_DEFAULT_PARAMETERS``; the values used are returned in
    "settings".
    """
    if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
        msg = (
            "Default EMG postprocessing expects processed EMG data from "
            "`preprocess_emg()`."
        )
        raise UnsupportedWorkflowError(msg)
    context: _PostprocessingContext = _build_context(
        envelope=processed_emg["envelope"],
        fs=processed_emg["fs"],
        peak_indices=np.asarray(peak_indices_from_events(events, processed_emg["fs"])),
        **kwargs,
    )
    _normalize_selected_postprocessing(context.selected_functions)
    _compute_baseline_default(context)
    _disable_unavailable_postprocessing(context)
    _ventilator_breaths_default(context)
    _detect_emg_events_default(context)
    _compute_emg_features_default(context)
    _compute_emg_quality_default(context)
    _compute_multimodal_quality_default(context)
    # TODO update settings
    return {
        "available": m3resp.emg.processing.core.available_postprocessing(),
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


def _disable_unavailable_postprocessing(context: _PostprocessingContext) -> None:

    if context.ventilator_signals is None:
        _disable_selected_function(
            context,
            "event_detection",
            "detect_ventilator_breath",
            "Needs ventilator volume input.",
        )
        _disable_selected_function(
            context,
            "event_detection",
            "find_occluded_breaths",
            "Needs ventilator pressure and PEEP inputs.",
        )
        _disable_selected_function(
            context,
            "quality_assessment",
            "detect_non_consecutive_manoeuvres",
            "Needs ventilator breath and manoeuvre indices.",
        )
        _disable_selected_function(
            context,
            "quality_assessment",
            "ventilator_respiratory_rate",
            "Needs ventilator volume input.",
        )

    if context.ecg_peak_indices is None or len(context.ecg_peak_indices) < 2:
        _disable_selected_function(
            context,
            "quality_assessment",
            "interpeak_distance",
            "Needs detected ECG peaks.",
        )

    if context.peep is None or np.isnan(context.peep):
        _disable_selected_function(
            context,
            "event_detection",
            "find_occluded_breaths",
            "Needs ventilator PEEP input.",
        )

    if context.baseline is None or len(context.peak_indices) < 2:
        # quality assessment:
        for function in [
            "evaluate_event_timing",
            "evaluate_respiratory_rates",
            "interpeak_distance",
            "detect_extreme_time_products",
            "evaluate_bell_curve_error",
            "percentage_under_baseline",
            "snr_pseudo",
        ]:
            _disable_selected_function(
                context,
                "quality_assessment",
                function,
                "Needs detected EMG breath peaks and baseline.",
            )

        # EVENT DETECTION
        for function in [
            "onoffpeak_baseline_crossing",
            "onoffpeak_slope_extrapolation",
        ]:
            _disable_selected_function(
                context,
                "event_detection",
                function,
                "Needs detected EMG breath peaks and baseline.",
            )
        # FEATURES
        for function in [
            "time_to_peak",
            "pseudo_slope",
            "time_product",
            "area_under_baseline",
            "amplitude",
        ]:
            _disable_selected_function(
                context,
                "features",
                function,
                "Needs detected EMG breath peaks and baseline.",
            )
    if len(context.peak_indices) < 2:
        _disable_selected_function(
            context, "features", "respiratory_rate", "Needs detected EMG breath peaks."
        )


class _DefaultsMixin:
    _preprocess_default = staticmethod(_preprocess_default)
    _postprocess_default = staticmethod(_postprocess_default)
    _detect_breaths_default = staticmethod(_detect_breaths_default)
    default_low_pass_hz = staticmethod(default_low_pass_hz)
    notch_max_frequency_hz = staticmethod(notch_max_frequency_hz)
    select_emg_channel = staticmethod(select_emg_channel)
    emg_envelope = staticmethod(emg_envelope)
