"""Registered ECG-artifact wavelet-denoising pipeline step."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.processing.windows import rolling_arv
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _record_step,
    _update_session_after_ecg_removal,
    _upstream_metadata,
)


@register_step(
    "emg.ecg_wavelet_denoising",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "ecg_peak_indices": "ecg_peak_indices",
    },
    writes=(
        "ecg_wavelet_cleaned_emg",
        "processed_emg_after_ecg",
        "ecg_wavelet_cleaned_signal",
        "wavelet_decomposition_result",
        "wavelet_thresholds_result",
        "wavelet_gate_mask_result",
    ),
    summary="Remove ECG peaks from EMG by a-trous wavelet shrinkage.",
    description="Remove ECG peaks from an EMG channel via a-trous wavelet shrinkage around each detected peak, via ReSurfEMGAdapter.wavelet_denoise_ecg.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.ecg_gating", "emg.ecg_estimated_subtraction"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'source' and 'fs'.",
        ),
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="ECG peak indices from 'emg.ecg_detect_peaks'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="source",
            value_type="string",
            default="filtered",
            description="Key into processed_emg to denoise.",
        ),
        StepParameter(
            name="hard_thresholding",
            value_type="boolean",
            default=True,
            description="Use hard (vs. soft) wavelet-coefficient thresholding.",
        ),
        StepParameter(
            name="levels",
            value_type="integer",
            default=4,
            minimum=1,
            description="Number of a-trous wavelet decomposition levels.",
        ),
        StepParameter(
            name="wavelet_type",
            value_type="string",
            default="db2",
            description="Wavelet family (PyWavelets name, e.g. 'db2').",
        ),
        StepParameter(
            name="fixed_threshold",
            value_type="number",
            default=4.5,
            minimum=0,
            description="Fixed wavelet-coefficient shrinkage threshold.",
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
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_wavelet_cleaned_emg",
            artifact_type="signal_array",
            description="Cleaned EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the cleaned signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ecg_wavelet_cleaned_signal",
            artifact_type="signal",
            description="Native Signal wrapping the cleaned EMG.",
        ),
        StepArtifact(
            name="wavelet_decomposition_result",
            artifact_type="parameter_result",
            axes=("level", "sample"),
            description="Native array-valued ParameterResult: wavelet decomposition coefficients.",
        ),
        StepArtifact(
            name="wavelet_thresholds_result",
            artifact_type="parameter_result",
            axes=("level", "sample"),
            description="Native array-valued ParameterResult: per-level thresholds applied.",
        ),
        StepArtifact(
            name="wavelet_gate_mask_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: boolean mask of denoised samples.",
        ),
    ),
)
def ecg_wavelet_denoising(
    session: M3Session,
    processed_emg: Any,
    ecg_peak_indices: Any,
    *,
    source: str = "filtered",
    hard_thresholding: bool = True,
    levels: int = 4,
    wavelet_type: str = "db2",
    fixed_threshold: float = 4.5,
    envelope_window_seconds: float | None = None,
) -> dict[str, Any]:
    if source not in processed_emg:
        raise ValueError(
            f"emg.ecg_wavelet_denoising source {source!r} is not present in "
            f"processed_emg; available keys: {sorted(processed_emg.keys())}."
        )

    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    original_length = len(array)
    padded_length = int(np.ceil(original_length / 2**levels) * 2**levels)

    cleaned, decomposition, thresholds, gate_mask = (
        session.emg_adapter.wavelet_denoise_ecg(
            array,
            ecg_peak_indices,
            sample_frequency=fs,
            hard_thresholding=hard_thresholding,
            levels=levels,
            wavelet_type=wavelet_type,
            fixed_threshold=fixed_threshold,
        )
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
        envelope_window_samples = max(1, int(effective_envelope_window_seconds * fs))
        envelope = rolling_arv(cleaned, window_length=envelope_window_samples)

    processed_emg_after_ecg = {
        **processed_emg,
        "filtered": cleaned,
        "envelope": envelope,
    }
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(cleaned), dtype=float) / fs
    wavelet_parameters = {
        "source": source,
        "hard_thresholding": hard_thresholding,
        "levels": levels,
        "wavelet_type": wavelet_type,
        "fixed_threshold": fixed_threshold,
        "original_length": original_length,
        "padded_length": padded_length,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
    }
    ecg_wavelet_cleaned_signal = Signal(
        values=cleaned,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_wavelet_cleaned",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="filtered",
        derived_from="processed",
        method="resurfemg.wavelet_denoising",
        metadata=dict(wavelet_parameters),
    )
    session.signals.add(ecg_wavelet_cleaned_signal)

    decomposition_result = ParameterResult(
        name="ecg_wavelet_decomposition",
        value=decomposition,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata={**wavelet_parameters, "axes": ["level", "sample"]},
    )
    thresholds_result = ParameterResult(
        name="ecg_wavelet_thresholds",
        value=thresholds,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata={**wavelet_parameters, "axes": ["level", "sample"]},
    )
    gate_mask_result = ParameterResult(
        name="ecg_wavelet_gate_mask",
        value=gate_mask,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata=dict(wavelet_parameters),
    )
    # All three are array-valued (decomposition/thresholds are 2D: level x
    # sample), so they reuse the shared parameter_result_arrays.npz exporter
    # (plan Phase 6.3) via session.export_summary() rather than a competing
    # EMG-specific array format.
    for array_result in (decomposition_result, thresholds_result, gate_mask_result):
        session.parameter_results.add(array_result)

    _record_step(
        session,
        "emg.ecg_wavelet_denoising",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.wavelet_denoising",
            operation="emg.ecg_wavelet_denoising",
            parameters=wavelet_parameters,
        ),
    )
    return {
        "ecg_wavelet_cleaned_emg": cleaned,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ecg_wavelet_cleaned_signal": ecg_wavelet_cleaned_signal,
        "wavelet_decomposition_result": decomposition_result,
        "wavelet_thresholds_result": thresholds_result,
        "wavelet_gate_mask_result": gate_mask_result,
    }
