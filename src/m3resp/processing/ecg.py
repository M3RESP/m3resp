"""ECG artifact removal from a one-dimensional EMG signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from m3resp.processing.filters import bandpass_filter
from m3resp.processing.windows import moving_average

OutputBandpassStage = Literal["before_subtraction", "after_subtraction"]


@dataclass(frozen=True)
class EstimatedECGSubtractionResult:
    """Signals and diagnostics produced by Estimated ECG Subtraction.

    Sample-index arrays refer to the input signal. ``qrs_indices`` has one
    row per template beat and three columns in Q, R, S order.
    """

    cleaned: np.ndarray
    estimated_ecg: np.ndarray
    detection_signal: np.ndarray
    dynamic_threshold: np.ndarray
    candidate_peak_indices: np.ndarray
    corrected_peak_indices: np.ndarray
    rejected_peak_indices: np.ndarray
    restored_peak_indices: np.ndarray
    template_peak_indices: np.ndarray
    qrs_indices: np.ndarray
    normalized_segments: np.ndarray
    normalized_template: np.ndarray
    template_sample_offsets: np.ndarray


def estimated_ecg_subtraction(
    values: np.ndarray,
    *,
    sample_frequency: float,
    detection_band_hz: tuple[float, float] = (4.0, 50.0),
    filter_order: int = 4,
    detection_smoothing_seconds: float = 0.0167,
    threshold_interval_seconds: float = 0.5,
    threshold_smoothing_seconds: float = 0.0125,
    qrs_window_seconds: float = 0.3,
    inter_qrs_tolerance: float = 0.66,
    minimum_template_beats: int = 3,
    minimum_qrs_interval_seconds: float | None = 0.25,
    maximum_qrs_interval_seconds: float | None = 2.0,
    output_bandpass_hz: tuple[float, float] | None = None,
    output_bandpass_stage: OutputBandpassStage = "after_subtraction",
    output_bandpass_order: int = 4,
) -> EstimatedECGSubtractionResult:
    """Estimate and subtract repeating ECG artifacts from contaminated EMG.

    This is a deterministic interpretation of the EES algorithm described by
    Jonkman et al. (2021). It does not need a separate ECG reference channel.
    The input should retain the ECG frequency content; an 80 Hz high-pass
    signal, for example, is not suitable for the paper's 4--50 Hz detection
    stage.

    The paper describes normalizing the smoothed detection signal to the
    amplitude of the input without defining the amplitude statistic. Here it
    is divided by the input peak-to-peak amplitude. This positive constant
    does not change threshold crossings because the dynamic threshold is
    calculated from the same normalized signal.

    The paper also does not define its inter-QRS outlier statistic. Here short
    intervals below ``(1 - inter_qrs_tolerance) * median_interval`` are treated
    as duplicate/false detections. Missing beats are restored at the strongest
    above-threshold detection peak in the expected interval.

    ``output_bandpass_hz`` is an optional extra bandpass applied exactly once,
    outside the QRS-detection stage; it is disabled by default and does not
    change any existing caller's output. QRS detection and the template are
    always built from the raw signal, regardless of ``output_bandpass_stage``.
    When set, ``output_bandpass_stage`` chooses which operand of the final
    subtraction is filtered: "before_subtraction" filters the raw signal
    immediately before subtracting the (unfiltered) estimated ECG, i.e.
    ``cleaned = bandpass(signal) - estimated_ecg``; "after_subtraction"
    subtracts first and filters the result, i.e.
    ``cleaned = bandpass(signal - estimated_ecg)``. These differ because
    bandpass filtering and subtracting the QRS template do not commute.
    """

    signal = _validate_input(
        values,
        sample_frequency=sample_frequency,
        detection_band_hz=detection_band_hz,
        filter_order=filter_order,
        detection_smoothing_seconds=detection_smoothing_seconds,
        threshold_interval_seconds=threshold_interval_seconds,
        threshold_smoothing_seconds=threshold_smoothing_seconds,
        qrs_window_seconds=qrs_window_seconds,
        inter_qrs_tolerance=inter_qrs_tolerance,
        minimum_template_beats=minimum_template_beats,
        minimum_qrs_interval_seconds=minimum_qrs_interval_seconds,
        maximum_qrs_interval_seconds=maximum_qrs_interval_seconds,
        output_bandpass_hz=output_bandpass_hz,
        output_bandpass_stage=output_bandpass_stage,
        output_bandpass_order=output_bandpass_order,
    )
    fs = float(sample_frequency)

    promoted_ecg = bandpass_filter(
        signal,
        cutoff_frequency=detection_band_hz,
        sample_frequency=fs,
        order=filter_order,
    )
    amplitude = float(np.ptp(signal))
    smoothed_rectified = moving_average(
        np.abs(promoted_ecg),
        window_size=_window_samples(detection_smoothing_seconds, fs),
    )
    detection_signal = smoothed_rectified / amplitude
    dynamic_threshold = _dynamic_midrange_threshold(
        detection_signal,
        interval_samples=_window_samples(threshold_interval_seconds, fs),
        smoothing_samples=_window_samples(threshold_smoothing_seconds, fs),
    )

    above_threshold = detection_signal > dynamic_threshold
    segments = _boolean_segments(above_threshold)
    candidate_peaks = np.asarray(
        [
            start + int(np.argmax(detection_signal[start:end]))
            for start, end in segments
            if end > start
        ],
        dtype=int,
    )
    if len(candidate_peaks) < minimum_template_beats:
        raise ValueError(
            "Estimated ECG Subtraction found too few potential QRS segments "
            f"({len(candidate_peaks)}); at least {minimum_template_beats} are needed."
        )

    corrected_peaks, rejected_peaks, restored_peaks = _correct_qrs_periodicity(
        candidate_peaks,
        detection_signal=detection_signal,
        dynamic_threshold=dynamic_threshold,
        tolerance=inter_qrs_tolerance,
    )
    template_peaks, qrs_indices, normalized_segments, beat_amplitudes = (
        _template_segments(
            signal,
            corrected_peaks,
            above_threshold=above_threshold,
            window_samples=_window_samples(qrs_window_seconds, fs),
        )
    )
    if len(qrs_indices) < minimum_template_beats:
        raise ValueError(
            "Estimated ECG Subtraction found too few complete QRS templates "
            f"({len(qrs_indices)}); at least {minimum_template_beats} are needed."
        )
    _validate_qrs_rate(
        qrs_indices[:, 1],
        sample_frequency=fs,
        minimum_interval_seconds=minimum_qrs_interval_seconds,
        maximum_interval_seconds=maximum_qrs_interval_seconds,
    )

    normalized_template = np.mean(normalized_segments, axis=0)
    estimated_ecg = _reconstruct_ecg(
        len(signal),
        qrs_indices=qrs_indices,
        beat_amplitudes=beat_amplitudes,
        normalized_template=normalized_template,
    )
    half_window = normalized_template.size // 2
    template_sample_offsets = np.arange(
        -half_window,
        normalized_template.size - half_window,
        dtype=int,
    )

    if output_bandpass_hz is not None and output_bandpass_stage == "before_subtraction":
        cleaned = (
            bandpass_filter(
                signal,
                cutoff_frequency=output_bandpass_hz,
                sample_frequency=fs,
                order=output_bandpass_order,
            )
            - estimated_ecg
        )
    else:
        cleaned = signal - estimated_ecg
        if (
            output_bandpass_hz is not None
            and output_bandpass_stage == "after_subtraction"
        ):
            cleaned = bandpass_filter(
                cleaned,
                cutoff_frequency=output_bandpass_hz,
                sample_frequency=fs,
                order=output_bandpass_order,
            )

    return EstimatedECGSubtractionResult(
        cleaned=cleaned,
        estimated_ecg=estimated_ecg,
        detection_signal=detection_signal,
        dynamic_threshold=dynamic_threshold,
        candidate_peak_indices=candidate_peaks,
        corrected_peak_indices=corrected_peaks,
        rejected_peak_indices=rejected_peaks,
        restored_peak_indices=restored_peaks,
        template_peak_indices=template_peaks,
        qrs_indices=qrs_indices,
        normalized_segments=normalized_segments,
        normalized_template=normalized_template,
        template_sample_offsets=template_sample_offsets,
    )


def _validate_input(
    values: np.ndarray,
    *,
    sample_frequency: float,
    detection_band_hz: tuple[float, float],
    filter_order: int,
    detection_smoothing_seconds: float,
    threshold_interval_seconds: float,
    threshold_smoothing_seconds: float,
    qrs_window_seconds: float,
    inter_qrs_tolerance: float,
    minimum_template_beats: int,
    minimum_qrs_interval_seconds: float | None,
    maximum_qrs_interval_seconds: float | None,
    output_bandpass_hz: tuple[float, float] | None = None,
    output_bandpass_stage: OutputBandpassStage = "after_subtraction",
    output_bandpass_order: int = 4,
) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1:
        raise ValueError("values must be a one-dimensional signal")
    if signal.size < 3:
        raise ValueError("values must contain at least three samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("values must not contain NaN or infinite values")
    if not np.isfinite(sample_frequency) or sample_frequency <= 0:
        raise ValueError("sample_frequency must be finite and positive")
    low, high = detection_band_hz
    if not (0 < low < high < sample_frequency / 2):
        raise ValueError(
            "detection_band_hz must be positive, increasing, and below Nyquist"
        )
    if filter_order < 1:
        raise ValueError("filter_order must be positive")
    durations = {
        "detection_smoothing_seconds": detection_smoothing_seconds,
        "threshold_interval_seconds": threshold_interval_seconds,
        "threshold_smoothing_seconds": threshold_smoothing_seconds,
        "qrs_window_seconds": qrs_window_seconds,
    }
    for name, duration in durations.items():
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= inter_qrs_tolerance < 1:
        raise ValueError("inter_qrs_tolerance must be between 0 and 1")
    if minimum_template_beats < 1:
        raise ValueError("minimum_template_beats must be positive")
    for name, interval in (
        ("minimum_qrs_interval_seconds", minimum_qrs_interval_seconds),
        ("maximum_qrs_interval_seconds", maximum_qrs_interval_seconds),
    ):
        if interval is not None and (not np.isfinite(interval) or interval <= 0):
            raise ValueError(f"{name} must be finite and positive or None")
    if (
        minimum_qrs_interval_seconds is not None
        and maximum_qrs_interval_seconds is not None
        and minimum_qrs_interval_seconds >= maximum_qrs_interval_seconds
    ):
        raise ValueError(
            "minimum_qrs_interval_seconds must be below maximum_qrs_interval_seconds"
        )
    if np.ptp(signal) <= np.finfo(float).eps:
        raise ValueError("values must have non-zero amplitude")
    if output_bandpass_hz is not None:
        output_low, output_high = output_bandpass_hz
        if not (0 < output_low < output_high < sample_frequency / 2):
            raise ValueError(
                "output_bandpass_hz must be positive, increasing, and below Nyquist"
            )
        if output_bandpass_order < 1:
            raise ValueError("output_bandpass_order must be positive")
        if output_bandpass_stage not in ("before_subtraction", "after_subtraction"):
            raise ValueError(
                "output_bandpass_stage must be 'before_subtraction' or "
                "'after_subtraction'"
            )
    return signal


def _validate_qrs_rate(
    r_indices: np.ndarray,
    *,
    sample_frequency: float,
    minimum_interval_seconds: float | None,
    maximum_interval_seconds: float | None,
) -> None:
    median_interval = float(np.median(np.diff(r_indices)) / sample_frequency)
    estimated_rate = 60 / median_interval
    if (
        minimum_interval_seconds is not None
        and median_interval < minimum_interval_seconds
    ):
        raise ValueError(
            "Estimated ECG Subtraction produced an implausibly short median "
            f"QRS interval ({median_interval:.3f} s, about {estimated_rate:.1f} "
            "beats/min). Review the detection signal and threshold, or lower "
            "minimum_qrs_interval_seconds only if this rate is expected."
        )
    if (
        maximum_interval_seconds is not None
        and median_interval > maximum_interval_seconds
    ):
        raise ValueError(
            "Estimated ECG Subtraction produced an implausibly long median "
            f"QRS interval ({median_interval:.3f} s, about {estimated_rate:.1f} "
            "beats/min). Review missed detections, or raise "
            "maximum_qrs_interval_seconds only if this rate is expected."
        )


def _window_samples(duration_seconds: float, sample_frequency: float) -> int:
    return max(1, round(duration_seconds * sample_frequency))


def _dynamic_midrange_threshold(
    values: np.ndarray,
    *,
    interval_samples: int,
    smoothing_samples: int,
) -> np.ndarray:
    centers: list[float] = []
    midranges: list[float] = []
    for start in range(0, len(values), interval_samples):
        end = min(start + interval_samples, len(values))
        block = values[start:end]
        centers.append((start + end - 1) / 2)
        midranges.append(float((np.min(block) + np.max(block)) / 2))
    sample_indices = np.arange(len(values), dtype=float)
    threshold = np.interp(sample_indices, centers, midranges)
    return moving_average(threshold, window_size=smoothing_samples)


def _boolean_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _correct_qrs_periodicity(
    candidate_peaks: np.ndarray,
    *,
    detection_signal: np.ndarray,
    dynamic_threshold: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.unique(np.asarray(candidate_peaks, dtype=int))
    median_interval = float(np.median(np.diff(candidates)))
    if median_interval <= 0:
        return candidates, np.empty(0, dtype=int), np.empty(0, dtype=int)

    minimum_interval = max(1.0, (1 - tolerance) * median_interval)
    accepted = [int(candidates[0])]
    rejected: list[int] = []
    for peak in candidates[1:]:
        if peak - accepted[-1] < minimum_interval:
            if detection_signal[peak] > detection_signal[accepted[-1]]:
                rejected.append(accepted[-1])
                accepted[-1] = int(peak)
            else:
                rejected.append(int(peak))
        else:
            accepted.append(int(peak))

    if len(accepted) > 1:
        median_interval = float(np.median(np.diff(accepted)))
    restored: list[int] = []
    corrected = [accepted[0]]
    for next_peak in accepted[1:]:
        while next_peak - corrected[-1] > (1 + tolerance) * median_interval:
            expected = corrected[-1] + median_interval
            radius = tolerance * median_interval
            start = max(corrected[-1] + 1, round(expected - radius))
            end = min(next_peak, round(expected + radius) + 1)
            if end <= start:
                break
            local = detection_signal[start:end]
            above = local > dynamic_threshold[start:end]
            if not np.any(above):
                break
            eligible = np.flatnonzero(above)
            inserted = start + int(eligible[np.argmax(local[eligible])])
            if inserted <= corrected[-1]:
                break
            corrected.append(inserted)
            restored.append(inserted)
        corrected.append(next_peak)

    return (
        np.asarray(corrected, dtype=int),
        np.asarray(rejected, dtype=int),
        np.asarray(restored, dtype=int),
    )


def _segment_containing(mask: np.ndarray, index: int) -> tuple[int, int]:
    start = int(index)
    end = int(index) + 1
    while start > 0 and mask[start - 1]:
        start -= 1
    while end < len(mask) and mask[end]:
        end += 1
    return start, end


# A beat whose Q-R or R-S scale sits far below the pack's typical scale is
# almost always a misplaced Q/S fiducial (e.g. landing one sample from R on a
# flat stretch) rather than a genuinely small QRS complex. Normalizing by
# such a near-zero scale blows the beat's normalized values up to whatever
# multiple of the pack's scale the local noise happens to be (seen in
# practice: a two-sample-apart Q/R pair with 1/2500th of the median scale
# produced a normalized peak of ~1900), and `np.mean` over
# `normalized_segments` then bakes that single outlier into the shared
# template used for every reconstructed beat. Rejecting scales below this
# fraction of the pack's median keeps such outliers out of the average.
_MINIMUM_RELATIVE_QRS_SCALE = 0.25


def _template_segments(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    *,
    above_threshold: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2

    candidates: list[tuple[int, int, int, int, int, float, float, float]] = []
    for detection_peak in peak_indices:
        segment_start, segment_end = _segment_containing(
            above_threshold, int(detection_peak)
        )
        if segment_end - segment_start < 3:
            continue
        r_index = segment_start + int(np.argmax(signal[segment_start:segment_end]))
        if r_index <= segment_start or r_index >= segment_end - 1:
            continue
        q_index = segment_start + int(np.argmin(signal[segment_start:r_index]))
        s_index = r_index + 1 + int(np.argmin(signal[r_index + 1 : segment_end]))
        start = r_index - half_window
        end = start + window_samples
        if start < 0 or end > len(signal):
            continue

        q_value = float(signal[q_index])
        r_value = float(signal[r_index])
        s_value = float(signal[s_index])
        qr_scale = r_value - q_value
        rs_scale = r_value - s_value
        if qr_scale <= np.finfo(float).eps or rs_scale <= np.finfo(float).eps:
            continue
        candidates.append(
            (
                int(detection_peak),
                q_index,
                r_index,
                s_index,
                start,
                end,
                qr_scale,
                rs_scale,
            )
        )

    qrs_rows: list[tuple[int, int, int]] = []
    template_peaks: list[int] = []
    normalized: list[np.ndarray] = []
    amplitudes: list[tuple[float, float, float]] = []
    if candidates:
        median_qr_scale = float(np.median([c[6] for c in candidates]))
        median_rs_scale = float(np.median([c[7] for c in candidates]))
        min_qr_scale = median_qr_scale * _MINIMUM_RELATIVE_QRS_SCALE
        min_rs_scale = median_rs_scale * _MINIMUM_RELATIVE_QRS_SCALE
    else:
        min_qr_scale = min_rs_scale = 0.0

    for (
        detection_peak,
        q_index,
        r_index,
        s_index,
        start,
        end,
        qr_scale,
        rs_scale,
    ) in candidates:
        if qr_scale < min_qr_scale or rs_scale < min_rs_scale:
            continue

        q_value = float(signal[q_index])
        r_value = float(signal[r_index])
        s_value = float(signal[s_index])
        beat = signal[start:end]
        relative_r = half_window
        normalized_beat = np.empty(window_samples, dtype=float)
        normalized_beat[: relative_r + 1] = (
            beat[: relative_r + 1] - q_value
        ) / qr_scale
        normalized_beat[relative_r:] = (beat[relative_r:] - s_value) / rs_scale
        qrs_rows.append((q_index, r_index, s_index))
        template_peaks.append(detection_peak)
        normalized.append(normalized_beat)
        amplitudes.append((q_value, r_value, s_value))

    return (
        np.asarray(template_peaks, dtype=int),
        np.asarray(qrs_rows, dtype=int).reshape(-1, 3),
        np.asarray(normalized, dtype=float).reshape(-1, window_samples),
        np.asarray(amplitudes, dtype=float).reshape(-1, 3),
    )


def _reconstruct_ecg(
    signal_length: int,
    *,
    qrs_indices: np.ndarray,
    beat_amplitudes: np.ndarray,
    normalized_template: np.ndarray,
) -> np.ndarray:
    estimated_sum = np.zeros(signal_length, dtype=float)
    contribution_count = np.zeros(signal_length, dtype=int)
    half_window = normalized_template.size // 2

    for (_, r_index, _), (q_value, r_value, s_value) in zip(
        qrs_indices, beat_amplitudes
    ):
        denormalized = np.empty_like(normalized_template)
        denormalized[: half_window + 1] = (
            normalized_template[: half_window + 1] * (r_value - q_value) + q_value
        )
        denormalized[half_window:] = (
            normalized_template[half_window:] * (r_value - s_value) + s_value
        )
        start = int(r_index) - half_window
        end = start + normalized_template.size
        estimated_sum[start:end] += denormalized
        contribution_count[start:end] += 1

    estimated = np.zeros(signal_length, dtype=float)
    covered = contribution_count > 0
    estimated[covered] = estimated_sum[covered] / contribution_count[covered]
    return estimated
