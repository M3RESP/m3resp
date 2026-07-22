"""Registered ECG-artifact estimated-subtraction pipeline step."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.core.events import Event
from m3resp.data import ParameterResult, Signal
from m3resp.processing.ecg import (
    OutputBandpassStage,
    estimated_ecg_subtraction as _estimated_ecg_subtraction,
)
from m3resp.processing.windows import rolling_arv
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _record_step,
    _update_session_after_ecg_removal,
    _upstream_metadata,
)


@register_step(
    "emg.ecg_estimated_subtraction",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=(
        "ees_cleaned_emg",
        "processed_emg_after_ecg",
        "ees_cleaned_signal",
        "ees_estimated_ecg_signal",
        "ees_detection_signal",
        "ees_dynamic_threshold_signal",
        "ees_qrs_events",
        "ees_r_peak_indices",
        "ees_candidate_peaks_result",
        "ees_corrected_peaks_result",
        "ees_rejected_peaks_result",
        "ees_restored_peaks_result",
        "ees_qrs_indices_result",
        "ees_normalized_segments_result",
        "ees_template_result",
    ),
    summary="Estimate and subtract ECG artifacts using a QRS template.",
    description=(
        "Native Estimated ECG Subtraction (EES): detects QRS beats directly in "
        "'source', builds a template, and subtracts the estimated ECG artifact "
        "from the EMG channel. An alternative to emg.ecg_gating/"
        "emg.ecg_wavelet_denoising that does not consume emg.ecg_detect_peaks. "
        "The candidate/corrected/rejected/restored QRS-index and template "
        "outputs are compatibility-only diagnostics for reviewing detected "
        "beats, not part of the native public result."
    ),
    category="preprocessing",
    modality="emg",
    optional_packages=(),
    alternatives=("emg.ecg_gating", "emg.ecg_wavelet_denoising"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'source' and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="source",
            value_type="string",
            default="filtered",
            description="Key into processed_emg to clean.",
        ),
        StepParameter(
            name="detection_low_hz",
            value_type="number",
            default=4.0,
            unit="Hz",
            description="Lower edge of the QRS-detection bandpass filter.",
        ),
        StepParameter(
            name="detection_high_hz",
            value_type="number",
            default=50.0,
            unit="Hz",
            description="Upper edge of the QRS-detection bandpass filter.",
        ),
        StepParameter(
            name="filter_order",
            value_type="integer",
            default=4,
            minimum=1,
            description="Detection bandpass filter order.",
        ),
        StepParameter(
            name="detection_smoothing_seconds",
            value_type="number",
            default=0.0167,
            unit="s",
            description="Smoothing applied to the detection signal.",
        ),
        StepParameter(
            name="threshold_interval_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            description="Interval over which the dynamic detection threshold is recomputed.",
        ),
        StepParameter(
            name="threshold_smoothing_seconds",
            value_type="number",
            default=0.0125,
            unit="s",
            description="Smoothing applied to the dynamic threshold.",
        ),
        StepParameter(
            name="qrs_window_seconds",
            value_type="number",
            default=0.3,
            unit="s",
            description="Window around each detected beat used to build the QRS template.",
        ),
        StepParameter(
            name="inter_qrs_tolerance",
            value_type="number",
            default=0.66,
            description="Fraction of the median inter-beat interval tolerated when validating beats.",
        ),
        StepParameter(
            name="minimum_template_beats",
            value_type="integer",
            default=3,
            minimum=1,
            description="Minimum number of beats required to build a stable template.",
        ),
        StepParameter(
            name="minimum_qrs_interval_seconds",
            value_type="number",
            required=False,
            default=0.25,
            unit="s",
            description="Shortest accepted inter-beat interval.",
        ),
        StepParameter(
            name="maximum_qrs_interval_seconds",
            value_type="number",
            required=False,
            default=2.0,
            unit="s",
            description="Longest accepted inter-beat interval.",
        ),
        StepParameter(
            name="envelope_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Envelope recomputation window on the cleaned signal. Defaults to the original preprocessing window.",
            advanced=True,
        ),
        StepParameter(
            name="output_bandpass_low_hz",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            description="Lower edge of an optional extra bandpass filter applied outside QRS detection. Disabled unless both low and high are set.",
            advanced=True,
        ),
        StepParameter(
            name="output_bandpass_high_hz",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            description="Upper edge of an optional extra bandpass filter applied outside QRS detection. Disabled unless both low and high are set.",
            advanced=True,
        ),
        StepParameter(
            name="output_bandpass_stage",
            value_type="string",
            default="after_subtraction",
            choices=("before_subtraction", "after_subtraction"),
            description="Which operand of the final subtraction gets filtered; detection and the template always use the raw signal either way. 'before_subtraction' filters the raw signal right before subtracting the (unfiltered) estimated ECG; 'after_subtraction' subtracts first and filters the result.",
            advanced=True,
        ),
        StepParameter(
            name="output_bandpass_order",
            value_type="integer",
            default=4,
            minimum=1,
            description="Filter order for the optional extra bandpass.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ees_cleaned_emg",
            artifact_type="signal_array",
            description="Cleaned (ECG-removed) EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the cleaned signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ees_cleaned_signal",
            artifact_type="signal",
            description="Native Signal wrapping the cleaned EMG.",
        ),
        StepArtifact(
            name="ees_estimated_ecg_signal",
            artifact_type="signal",
            description="Native Signal wrapping the estimated ECG artifact that was subtracted.",
        ),
        StepArtifact(
            name="ees_detection_signal",
            artifact_type="signal",
            description="Native Signal of the QRS-detection signal used to find beats.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_dynamic_threshold_signal",
            artifact_type="signal",
            description="Native Signal of the dynamic detection threshold over time.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_r_peak_indices",
            artifact_type="index_array",
            description="Detected R-peak sample indices.",
        ),
        StepArtifact(
            name="ees_qrs_events",
            artifact_type="event_list",
            description="Native Event per detected QRS beat.",
        ),
        StepArtifact(
            name="ees_candidate_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: initially detected candidate peak indices.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_corrected_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: candidate peaks after correction.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_rejected_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: peaks rejected during correction.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_restored_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: peaks restored by periodicity.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_qrs_indices_result",
            artifact_type="parameter_result",
            axes=("beat", "wave_q_r_s"),
            description="Native array-valued ParameterResult: Q/R/S sample index per detected beat.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_normalized_segments_result",
            artifact_type="parameter_result",
            axes=("beat", "template_sample"),
            description="Native array-valued ParameterResult: each beat's segment normalized onto the template timebase.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_template_result",
            artifact_type="parameter_result",
            axes=("template_sample",),
            description="Native array-valued ParameterResult: the fitted QRS template.",
            compatibility_only=True,
        ),
    ),
)
def ecg_estimated_subtraction(
    session: M3Session,
    processed_emg: Any,
    *,
    source: str = "filtered",
    detection_low_hz: float = 4.0,
    detection_high_hz: float = 50.0,
    filter_order: int = 4,
    detection_smoothing_seconds: float = 0.0167,
    threshold_interval_seconds: float = 0.5,
    threshold_smoothing_seconds: float = 0.0125,
    qrs_window_seconds: float = 0.3,
    inter_qrs_tolerance: float = 0.66,
    minimum_template_beats: int = 3,
    minimum_qrs_interval_seconds: float | None = 0.25,
    maximum_qrs_interval_seconds: float | None = 2.0,
    envelope_window_seconds: float | None = None,
    output_bandpass_low_hz: float | None = None,
    output_bandpass_high_hz: float | None = None,
    output_bandpass_stage: OutputBandpassStage = "after_subtraction",
    output_bandpass_order: int = 4,
) -> dict[str, Any]:
    """Run the paper-based Estimated ECG Subtraction method.

    The step detects ECG directly in ``source``; it does not consume the
    output of ``emg.ecg_detect_peaks``. Diagnostic signals and arrays are
    retained so the detected beats and estimated artifact can be reviewed.
    """

    if source not in processed_emg:
        raise ValueError(
            f"emg.ecg_estimated_subtraction source {source!r} is not present in "
            f"processed_emg; available keys: {sorted(processed_emg.keys())}."
        )
    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    output_bandpass_hz = (
        (output_bandpass_low_hz, output_bandpass_high_hz)
        if output_bandpass_low_hz is not None and output_bandpass_high_hz is not None
        else None
    )
    result = _estimated_ecg_subtraction(
        array,
        sample_frequency=fs,
        detection_band_hz=(detection_low_hz, detection_high_hz),
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

    original_window_seconds = (processed_emg.get("filter") or {}).get(
        "envelope_window_seconds"
    )
    effective_envelope_window_seconds = (
        envelope_window_seconds
        if envelope_window_seconds is not None
        else original_window_seconds
    )
    envelope = processed_emg.get("envelope")
    if effective_envelope_window_seconds is not None:
        envelope = rolling_arv(
            result.cleaned,
            window_length=max(1, int(effective_envelope_window_seconds * fs)),
        )
    processed_emg_after_ecg = {
        **processed_emg,
        "filtered": result.cleaned,
        "envelope": envelope,
    }
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(array), dtype=float) / fs
    method = "m3resp.estimated_ecg_subtraction"
    parameters = {
        "source": source,
        "detection_low_hz": detection_low_hz,
        "detection_high_hz": detection_high_hz,
        "filter_order": filter_order,
        "detection_smoothing_seconds": detection_smoothing_seconds,
        "threshold_interval_seconds": threshold_interval_seconds,
        "threshold_smoothing_seconds": threshold_smoothing_seconds,
        "qrs_window_seconds": qrs_window_seconds,
        "inter_qrs_tolerance": inter_qrs_tolerance,
        "minimum_template_beats": minimum_template_beats,
        "minimum_qrs_interval_seconds": minimum_qrs_interval_seconds,
        "maximum_qrs_interval_seconds": maximum_qrs_interval_seconds,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
        "output_bandpass_low_hz": output_bandpass_low_hz,
        "output_bandpass_high_hz": output_bandpass_high_hz,
        "output_bandpass_stage": output_bandpass_stage,
        "output_bandpass_order": output_bandpass_order,
    }
    signal_metadata = dict(parameters)
    ees_cleaned_signal = Signal(
        values=result.cleaned,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_ees_cleaned",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="filtered",
        derived_from="processed",
        method=method,
        metadata=signal_metadata,
    )
    ees_estimated_ecg_signal = Signal(
        values=result.estimated_ecg,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_estimated_ecg",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    ees_detection_signal = Signal(
        values=result.detection_signal,
        time=time,
        sample_frequency=fs,
        unit=None,
        name=f"{label}_ees_detection",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    ees_dynamic_threshold_signal = Signal(
        values=result.dynamic_threshold,
        time=time,
        sample_frequency=fs,
        unit=None,
        name=f"{label}_ees_dynamic_threshold",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    for output_signal in (
        ees_cleaned_signal,
        ees_estimated_ecg_signal,
        ees_detection_signal,
        ees_dynamic_threshold_signal,
    ):
        session.signals.add(output_signal)

    restored = set(result.restored_peak_indices.tolist())
    ees_qrs_events = [
        Event(
            name="ees_qrs",
            modality="emg",
            time=float(r_index) / fs,
            sample_index=int(r_index),
            metadata={
                **parameters,
                "q_sample_index": int(q_index),
                "s_sample_index": int(s_index),
                "detection_peak_sample_index": int(detection_peak),
                "restored_by_periodicity": int(detection_peak) in restored,
            },
        )
        for detection_peak, (q_index, r_index, s_index) in zip(
            result.template_peak_indices, result.qrs_indices
        )
    ]
    session.add_events("ees_qrs", ees_qrs_events)

    result_specs = (
        ("ees_candidate_peaks", result.candidate_peak_indices, ["candidate"]),
        ("ees_corrected_peaks", result.corrected_peak_indices, ["corrected"]),
        ("ees_rejected_peaks", result.rejected_peak_indices, ["rejected"]),
        ("ees_restored_peaks", result.restored_peak_indices, ["restored"]),
        ("ees_qrs_indices", result.qrs_indices, ["beat", "wave_q_r_s"]),
        (
            "ees_normalized_segments",
            result.normalized_segments,
            ["beat", "template_sample"],
        ),
        ("ees_normalized_template", result.normalized_template, ["template_sample"]),
    )
    parameter_results: dict[str, ParameterResult] = {}
    for name, values, axes in result_specs:
        parameter_result = ParameterResult(
            name=name,
            value=values,
            modality="emg",
            channel=label,
            method=method,
            metadata={
                **parameters,
                "axes": axes,
                "template_sample_offsets": (
                    result.template_sample_offsets.tolist()
                    if "template" in name
                    else None
                ),
            },
        )
        session.parameter_results.add(parameter_result)
        parameter_results[name] = parameter_result

    _record_step(
        session,
        "emg.ecg_estimated_subtraction",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.ecg.estimated_ecg_subtraction",
            operation="emg.ecg_estimated_subtraction",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.ecg",
        ),
    )
    return {
        "ees_cleaned_emg": result.cleaned,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ees_cleaned_signal": ees_cleaned_signal,
        "ees_estimated_ecg_signal": ees_estimated_ecg_signal,
        "ees_detection_signal": ees_detection_signal,
        "ees_dynamic_threshold_signal": ees_dynamic_threshold_signal,
        "ees_qrs_events": ees_qrs_events,
        "ees_r_peak_indices": result.qrs_indices[:, 1],
        "ees_candidate_peaks_result": parameter_results["ees_candidate_peaks"],
        "ees_corrected_peaks_result": parameter_results["ees_corrected_peaks"],
        "ees_rejected_peaks_result": parameter_results["ees_rejected_peaks"],
        "ees_restored_peaks_result": parameter_results["ees_restored_peaks"],
        "ees_qrs_indices_result": parameter_results["ees_qrs_indices"],
        "ees_normalized_segments_result": parameter_results["ees_normalized_segments"],
        "ees_template_result": parameter_results["ees_normalized_template"],
    }
