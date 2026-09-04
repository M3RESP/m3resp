"""Registered ECG-artifact gating pipeline step."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.processing.windows import ENVELOPE_METHODS, rolling_envelope
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _record_step,
    _update_session_after_ecg_removal,
    _upstream_metadata,
    resolve_emg_source,
)


def _build_gate_mask(
    n_samples: int, peak_indices: Any, *, gate_width_samples: int, fill_method: int
) -> np.ndarray:
    """A boolean mask marking the (clipped-to-bounds) gated region around
    each peak. Purely descriptive - it reports which samples the cleaned
    array had replaced, and is never fed back into it.

    The blanked region depends on the fill method. ReSurfEMG's RMS fill
    (method 3) spans ``int(peak +/- gate_width / 2)`` while the zero,
    interpolation and prior-segment fills (methods 0, 1, 2) span
    ``peak +/- gate_width // 2``. On an odd gate width - the 205-sample
    default among them - the RMS fill starts one sample earlier. The
    arithmetic below mirrors each case so the mask names exactly the samples
    that were replaced."""

    mask = np.zeros(n_samples, dtype=bool)
    for peak in peak_indices:
        if fill_method == 3:
            start = int(int(peak) - gate_width_samples / 2)
            end = int(int(peak) + gate_width_samples / 2)
        else:
            half_width = gate_width_samples // 2
            start = int(peak) - half_width
            end = int(peak) + half_width
        mask[max(0, start) : min(n_samples, end)] = True
    return mask


@register_step(
    "emg.ecg_gating",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "ecg_peak_indices": "ecg_peak_indices",
    },
    writes=(
        "ecg_gated_emg",
        "processed_emg_after_ecg",
        "ecg_gated_signal",
        "ecg_gate_mask_result",
    ),
    summary="Remove ECG peaks from EMG by gating (zero/interpolate/replace).",
    description="Remove ECG peaks from an EMG channel by gating each detected peak (zero/interpolate/replace), via ReSurfEMGAdapter.gate_ecg.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=(
        "session.processed.emg",
        "session.signals",
        "session.parameter_results",
    ),
    alternatives=("emg.ecg_wavelet_denoising", "emg.ecg_estimated_subtraction"),
    mutually_exclusive_parameters=(("gate_width_seconds", "gate_width_samples"),),
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
            required=False,
            default=None,
            description="Key into processed_emg to gate. Defaults to the most-processed trace present: the ECG-cleaned signal when an earlier removal step produced one, otherwise the band-passed signal.",
        ),
        StepParameter(
            name="gate_width_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Gate width in seconds. Mutually exclusive with 'gate_width_samples'.",
        ),
        StepParameter(
            name="gate_width_samples",
            value_type="integer",
            required=False,
            default=None,
            minimum=1,
            description="Gate width in samples. Mutually exclusive with 'gate_width_seconds'. Defaults to 205 samples (resurfemg's own default) when both are unset.",
        ),
        StepParameter(
            name="fill_method",
            value_type="integer",
            default=1,
            choices=(0, 1, 2, 3),
            description="Gate fill strategy: 0 zeros, 1 interpolation, 2 mean of a neighboring segment, 3 running-RMS-based replacement.",
        ),
        StepParameter(
            name="envelope_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Envelope recomputation window on the gated signal. Defaults to the original preprocessing window.",
            advanced=True,
        ),
        StepParameter(
            name="envelope_method",
            value_type="string",
            required=False,
            default=None,
            choices=ENVELOPE_METHODS,
            description="Envelope method for the recomputation on the gated signal. Defaults to the method preprocessing used, so the two cannot disagree.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_gated_emg",
            artifact_type="signal_array",
            description="Gated EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the gated signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ecg_gated_signal",
            artifact_type="signal",
            description="Native Signal wrapping the gated EMG.",
        ),
        StepArtifact(
            name="ecg_gate_mask_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: boolean mask of gated samples.",
        ),
    ),
)
def ecg_gating(
    session: M3Session,
    processed_emg: Any,
    ecg_peak_indices: Any,
    *,
    source: str | None = None,
    gate_width_seconds: float | None = None,
    gate_width_samples: int | None = None,
    fill_method: int = 1,
    envelope_window_seconds: float | None = None,
    envelope_method: str | None = None,
) -> dict[str, Any]:
    """Remove ECG peaks from an EMG channel by gating each detected peak (zero/interpolate/replace), via ReSurfEMGAdapter.gate_ecg.

    Args:
        session (M3Session): The M3Session object.
        processed_emg (Any): The processed EMG bundle supplying 'source' and 'fs'.
        ecg_peak_indices (Any): The ECG peak indices from 'emg.ecg_detect_peaks'.
        source (str): Key into processed_emg to gate.
        gate_width_seconds (float | None): Gate width in seconds. Mutually exclusive with 'gate_width_samples'.
        gate_width_samples (int | None): Gate width in samples. Mutually exclusive with 'gate_width_seconds'.
        fill_method (int): Gate fill strategy.
        envelope_window_seconds (float | None): Envelope recomputation window on the gated signal.
        envelope_method (str | None): {'rms', 'arv'}, envelope method for the recomputation. Defaults to the method preprocessing used.

    Returns:
        ecg_gated_emg (signal_array): Gated EMG array.
        processed_emg_after_ecg (emg_processed_bundle): Updated processed-EMG bundle with the gated signal as its 'filtered'/'envelope'.
        ecg_gated_signal (signal): Native Signal wrapping the gated EMG.
        ecg_gate_mask_result (parameter_result): Native array-valued ParameterResult: boolean mask of gated samples.
    """
    if gate_width_seconds is not None and gate_width_samples is not None:
        raise ValueError(
            "emg.ecg_gating: set only one of gate_width_seconds or "
            "gate_width_samples, not both."
        )
    source = resolve_emg_source(processed_emg, source, "emg.ecg_gating")

    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    if gate_width_seconds is not None:
        effective_gate_width_samples = max(1, int(gate_width_seconds * fs))
    elif gate_width_samples is not None:
        effective_gate_width_samples = gate_width_samples
    else:
        effective_gate_width_samples = 205  # resurfemg's own default

    gated = session.emg_adapter.gate_ecg(
        array,
        ecg_peak_indices,
        gate_width_samples=effective_gate_width_samples,
        fill_method=fill_method,
    )
    gate_mask = _build_gate_mask(
        len(array),
        ecg_peak_indices,
        gate_width_samples=effective_gate_width_samples,
        fill_method=fill_method,
    )

    original_filter = processed_emg.get("filter") or {}
    original_window_seconds = original_filter.get("envelope_window_seconds")
    effective_envelope_window_seconds = (
        envelope_window_seconds
        if envelope_window_seconds is not None
        else original_window_seconds
    )
    # Default to whatever method preprocessing used, so the recomputed
    # envelope is the same kind of envelope as the one it replaces; only fall
    # back to the "rms" default for a bundle that predates the field.
    effective_envelope_method = (
        envelope_method
        if envelope_method is not None
        else original_filter.get("envelope_method") or "rms"
    )
    envelope = processed_emg.get("envelope")
    if effective_envelope_window_seconds is not None:
        envelope_window_samples = max(1, int(effective_envelope_window_seconds * fs))
        envelope = rolling_envelope(
            gated,
            window_length=envelope_window_samples,
            method=effective_envelope_method,
        )

    processed_emg_after_ecg = {
        **processed_emg,
        # Band-passing and gating are separate steps: "filtered" keeps the
        # band-passed signal and the gated one lands beside it, so what
        # gating did stays visible and recomputable.
        "ecg_cleaned": gated,
        "envelope": envelope,
        # Carry the *effective* envelope settings forward, so a later
        # recomputation off this bundle reuses what was actually applied here
        # rather than the pre-gating preprocessing values.
        "filter": {
            **original_filter,
            "envelope_window_seconds": effective_envelope_window_seconds,
            "envelope_method": effective_envelope_method,
        },
    }
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(gated), dtype=float) / fs
    gating_parameters = {
        "source": source,
        "requested_gate_width_seconds": gate_width_seconds,
        "requested_gate_width_samples": gate_width_samples,
        "effective_gate_width_samples": effective_gate_width_samples,
        "fill_method": fill_method,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
        "effective_envelope_method": effective_envelope_method,
    }
    ecg_gated_signal = Signal(
        values=gated,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_gated",
        modality="emg",
        category="electrical_potential",
        channel=label,
        source="resurfemg",
        processing_state="intermediate",
        derived_from="processed",
        method="resurfemg.gating",
        metadata=dict(gating_parameters),
    )
    session.signals.add(ecg_gated_signal)

    gate_mask_result = ParameterResult(
        name="ecg_gate_mask",
        value=gate_mask,
        modality="emg",
        channel=label,
        method="resurfemg.gating",
        metadata=dict(gating_parameters),
    )
    # Array-valued, so this reuses the shared parameter_result_arrays.npz
    # exporter (plan Phase 6.3) rather than a competing EMG-specific one -
    # session.export_summary() already routes any array-valued
    # ParameterResult there.
    session.parameter_results.add(gate_mask_result)

    _record_step(
        session,
        "emg.ecg_gating",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.gating",
            operation="emg.ecg_gating",
            parameters=gating_parameters,
        ),
    )
    return {
        "ecg_gated_emg": gated,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ecg_gated_signal": ecg_gated_signal,
        "ecg_gate_mask_result": gate_mask_result,
    }
