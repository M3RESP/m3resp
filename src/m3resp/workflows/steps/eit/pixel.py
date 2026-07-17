"""Registered pixel-level EIT pipeline steps."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _EITPROCESSING,
    _SESSION_ARTIFACT,
    _record_step,
    _upstream_metadata,
)


def _object_array_to_float(values: Any) -> np.ndarray:
    """Convert an object-dtype array using `None` for missing entries into a
    float array with NaN in place of `None`, preserving its shape."""

    array = np.asarray(values, dtype=object)
    flat = [np.nan if value is None else float(value) for value in array.ravel()]
    return np.array(flat, dtype=float).reshape(array.shape)


@register_step(
    "eit.pixel_tiv",
    reads={
        "filtered_eit": "filtered_eit",
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
        "session": "session",
    },
    writes=("pixel_tiv", "pixel_tiv_result"),
    summary="Compute per-pixel tidal impedance variation (TIV).",
    description="Compute per-breath, per-pixel tidal impedance variation via eitprocessing's TIV.compute_pixel_parameter.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="filtered_eit",
            artifact_type="eit_pixel_signal",
            description="Filtered EIT pixel signal to compute per-pixel TIV on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying breath timing.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured breath detector from 'eit.detect_breaths'.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="tiv_timing",
            value_type="choice",
            default="continuous",
            choices=("pixel", "continuous"),
            description="Whether breath timing is taken per-pixel or from the continuous (global) signal.",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_tivs",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_tiv",
            artifact_type="eit_sparse_data",
            description="Per-breath, per-pixel TIV values (upstream SparseData).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pixel_tiv_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult, shape (breath, row, column).",
            axes=("breath", "row", "column"),
        ),
    ),
)
def pixel_tiv(
    filtered_eit: Any,
    signal: Any,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
    *,
    tiv_timing: Literal["pixel", "continuous"] = "continuous",
    result_label: str = "pixel_tivs",
) -> dict[str, Any]:
    result = session.eit_adapter.compute_pixel_tiv(
        filtered_eit,
        signal,
        sequence=eit_sequence,
        breath_detector=breath_detector,
        tiv_timing=tiv_timing,
        result_label=result_label,
    )

    values = _object_array_to_float(result.values)
    time = _object_array_to_float(result.time)
    valid_breath_indices = [
        index for index in range(values.shape[0]) if not np.all(np.isnan(values[index]))
    ]

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.parameters.tidal_impedance_variation."
            "TIV.compute_pixel_parameter"
        ),
        operation="eit.pixel_tiv",
        parameters={"tiv_timing": tiv_timing, "result_label": result_label},
    )
    parameter_metadata = dict(metadata)
    parameter_metadata.update(
        {
            "time": time.tolist(),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "axes": ["breath", "row", "column"],
            "tiv_timing": tiv_timing,
            "result_label": result_label,
            "valid_breath_indices": valid_breath_indices,
            "valid_breath_fraction": (
                len(valid_breath_indices) / values.shape[0] if values.shape[0] else 0.0
            ),
        }
    )
    pixel_tiv_result = ParameterResult(
        name=getattr(result, "label", None) or result_label,
        value=values,
        modality="eit",
        unit=getattr(result, "unit", None),
        method="eitprocessing.TIV",
        metadata=parameter_metadata,
    )
    session.parameter_results.add(pixel_tiv_result)

    _record_step(session, "eit.pixel_tiv", metadata=metadata)
    return {"pixel_tiv": result, "pixel_tiv_result": pixel_tiv_result}


_ALLOWED_PIXEL_BREATH_PHASE_MODES = {"negative amplitude", "phase shift", "none", None}


def _pixel_breaths_to_landmark_array(values: Any) -> np.ndarray:
    """Convert `PixelBreath`'s `(breath, row, column)` object array of
    `Breath | None` into a numeric `(breath, row, column, landmark)` array,
    where landmark is `[start_time, middle_time, end_time]`. Missing pixel
    breaths (including the always-unavailable first/last global breath)
    become NaN."""

    array = np.asarray(values, dtype=object)
    n_breaths, n_rows, n_cols = array.shape
    landmarks = np.full((n_breaths, n_rows, n_cols, 3), np.nan, dtype=float)
    for breath_index in range(n_breaths):
        for row in range(n_rows):
            for col in range(n_cols):
                breath = array[breath_index, row, col]
                if breath is not None:
                    landmarks[breath_index, row, col] = (
                        breath.start_time,
                        breath.middle_time,
                        breath.end_time,
                    )
    return landmarks


@register_step(
    "eit.pixel_breaths",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "eit_sequence": "eit_sequence",
        "session": "session",
    },
    writes=("pixel_breaths", "pixel_breath_timing_result"),
    summary="Detect per-pixel breath timing (start/middle/end of in-/deflation).",
    description="Detect per-pixel breath start/middle/end timing via eitprocessing's PixelBreath.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to detect pixel breaths on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying overall breath timing.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="phase_correction_mode",
            value_type="choice",
            required=False,
            default="negative amplitude",
            choices=("negative amplitude", "phase shift", "none"),
            description="Per-pixel phase correction method. Null is also accepted, equivalent to 'none'.",
        ),
        StepParameter(
            name="minimum_duration_seconds",
            value_type="number",
            default=2 / 3,
            unit="s",
            minimum=0,
            description="Minimum per-pixel breath duration accepted.",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_breaths",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_breaths",
            artifact_type="eit_sparse_data",
            description="Per-pixel breath objects (upstream SparseData of Breath | None).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pixel_breath_timing_result",
            artifact_type="parameter_result",
            unit="s",
            description="Native array-valued ParameterResult of [start, middle, end] landmark times.",
            axes=("breath", "row", "column", "landmark"),
        ),
    ),
)
def pixel_breaths(
    eit_data: Any,
    timing_data: Any,
    eit_sequence: Any,
    session: M3Session,
    *,
    phase_correction_mode: Literal["negative amplitude", "phase shift", "none"]
    | None = "negative amplitude",
    minimum_duration_seconds: float = 2 / 3,
    result_label: str = "pixel_breaths",
) -> dict[str, Any]:
    if phase_correction_mode not in _ALLOWED_PIXEL_BREATH_PHASE_MODES:
        raise ValueError(
            "eit.pixel_breaths 'phase_correction_mode' must be one of "
            "'negative amplitude', 'phase shift', 'none', or null; "
            f"got {phase_correction_mode!r}."
        )

    result = session.eit_adapter.find_pixel_breaths(
        eit_data,
        timing_data,
        sequence=eit_sequence,
        phase_correction_mode=phase_correction_mode,
        minimum_duration_seconds=minimum_duration_seconds,
        result_label=result_label,
    )

    landmarks = _pixel_breaths_to_landmark_array(result.values)
    valid = ~np.isnan(landmarks[..., 0])

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.features.pixel_breath.PixelBreath.find_pixel_breaths"
        ),
        operation="eit.pixel_breaths",
        parameters={
            "phase_correction_mode": phase_correction_mode,
            "minimum_duration_seconds": minimum_duration_seconds,
            "result_label": result_label,
        },
    )
    metadata.update(
        {
            "shape": list(landmarks.shape),
            "dtype": str(landmarks.dtype),
            "axes": ["breath", "row", "column", "landmark"],
            "landmarks": ["start_time", "middle_time", "end_time"],
            "valid_pixel_breath_count": int(valid.sum()),
            "valid_pixel_breath_fraction": float(valid.mean()) if valid.size else 0.0,
        }
    )
    pixel_breath_timing_result = ParameterResult(
        name=result_label,
        value=landmarks,
        modality="eit",
        unit="s",
        method="eitprocessing.PixelBreath",
        metadata=metadata,
    )
    session.parameter_results.add(pixel_breath_timing_result)

    _record_step(session, "eit.pixel_breaths", metadata=metadata)
    return {
        "pixel_breaths": result,
        "pixel_breath_timing_result": pixel_breath_timing_result,
    }
