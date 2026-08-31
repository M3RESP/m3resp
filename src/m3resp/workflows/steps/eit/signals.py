"""Registered EIT global-signal pipeline steps (impedance/TIV/EELI)."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
)
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


@register_step(
    "eit.global_impedance",
    reads={"signal": "filtered_eit", "eit_sequence": "eit_sequence"},
    writes=("global_impedance",),
    summary="Compute and store the summed (global) impedance of an EIT signal.",
    description="Sum pixel impedance across the image to produce the single-channel global impedance waveform used for breath detection.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to sum.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the summed signal is added onto.",
            public=False,
            compatibility_only=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="global_impedance",
            artifact_type="eit_global_impedance",
            description="Summed (global) impedance waveform.",
            compatibility_only=True,
        ),
    ),
)
def global_impedance(
    signal: Any,
    *,
    eit_sequence: Any,
) -> dict[str, Any]:
    summed = signal.get_summed_impedance()
    add_to_collection(eit_sequence.continuous_data, summed)
    return {"global_impedance": summed}


@register_step(
    "eit.detect_breaths",
    reads={"signal": "detection_signal"},
    writes=("breath_intervals", "breath_detector"),
    summary="Detect breaths on a continuous EIT impedance signal.",
    description="Detect breath start/end times on the global impedance waveform via eitprocessing's BreathDetection.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="detection_signal",
            description="Continuous impedance signal (typically the detection-window slice) to detect breaths on.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="min_duration_s",
            value_type="number",
            default=2 / 3,
            unit="s",
            minimum=0,
            description="Minimum breath duration accepted by the detector.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="breath_intervals",
            artifact_type="interval_collection",
            description="Detected breath intervals.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured upstream BreathDetection instance, reusable by later steps.",
            public=False,
            compatibility_only=True,
        ),
    ),
)
def detect_breaths(signal: Any, *, min_duration_s: float = 2 / 3) -> dict[str, Any]:
    from eitprocessing.features.breath_detection import BreathDetection

    detector = BreathDetection(minimum_duration=min_duration_s)
    return {
        "breath_intervals": detector.find_breaths(signal),
        "breath_detector": detector,
    }


@register_step(
    "eit.normalize_breaths",
    reads={"breath_intervals": "breath_intervals", "session": "session"},
    writes=(),
    summary="Normalize detected EIT breath intervals into session events.",
    description="Convert detected EIT breath intervals into native BreathEvents and store them on the session as 'eit_breaths'.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    parameters_reviewed=True,
    session_writes=("session.events.eit_breaths",),
    input_artifacts=(
        StepArtifact(
            name="breath_intervals",
            artifact_type="interval_collection",
            default_context_key="breath_intervals",
            description="Breath intervals from 'eit.detect_breaths' to normalize.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
)
def normalize_breaths(
    breath_intervals: Any,
    *,
    session: M3Session,
) -> dict[str, Any]:
    events = session.eit_adapter.detect_breaths({"breath_intervals": breath_intervals})
    session.add_events("eit_breaths", events)
    return {}


@register_step(
    "eit.continuous_tiv",
    reads={
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
    },
    writes=("continuous_tiv",),
    summary="Compute continuous tidal impedance variation (TIV).",
    description="Compute per-breath tidal impedance variation on the global impedance waveform via eitprocessing's TIV.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform to compute TIV on.",
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
    ),
    output_artifacts=(
        StepArtifact(
            name="continuous_tiv",
            artifact_type="eit_sparse_data",
            description="Per-breath continuous TIV values (upstream SparseData).",
            compatibility_only=True,
        ),
    ),
)
def continuous_tiv(
    signal: Any,
    *,
    eit_sequence: Any,
    breath_detector: Any,
) -> dict[str, Any]:
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(breath_detection=breath_detector).compute_parameter(
        signal, sequence=eit_sequence, store=False, result_label="continuous_tivs"
    )
    add_to_collection(eit_sequence.sparse_data, result)
    return {"continuous_tiv": result}


def _sparse_data_to_array_parameter(
    obj: Any, *, modality: str, method: str, metadata: dict[str, Any]
) -> ParameterResult:
    """Convert an `eitprocessing.SparseData`-shaped object (one value per
    breath) into a single array-valued `ParameterResult`, preserving upstream
    time coordinates and unit rather than exploding it into one result per
    breath."""

    values = np.asarray(obj.values, dtype=float)
    metadata = dict(metadata)
    metadata["time"] = np.asarray(obj.time, dtype=float).tolist()
    metadata["shape"] = list(values.shape)
    metadata["dtype"] = str(values.dtype)
    name = getattr(obj, "label", None) or getattr(obj, "name", None) or "parameter"
    return ParameterResult(
        name=name,
        value=values,
        modality=modality,
        unit=getattr(obj, "unit", None),
        method=method,
        metadata=metadata,
    )


@register_step(
    "eit.eeli",
    reads={
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
        "session": "session",
    },
    writes=("eeli", "eeli_result"),
    summary="Compute end-expiratory lung impedance (EELI).",
    description="Compute per-breath end-expiratory lung impedance (EELI) via eitprocessing's EELI parameter.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.parameter_results",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform to compute EELI on.",
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
            name="result_label",
            value_type="string",
            default="continuous_eelis",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="eeli",
            artifact_type="eit_sparse_data",
            description="Per-breath EELI values (upstream SparseData).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eeli_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult for EELI, one value per breath.",
        ),
    ),
)
def eeli(
    signal: Any,
    *,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
    result_label: str = "continuous_eelis",
) -> dict[str, Any]:
    result = session.eit_adapter.compute_eeli(
        signal,
        sequence=eit_sequence,
        breath_detector=breath_detector,
        result_label=result_label,
    )
    metadata = _upstream_metadata(
        source_function="eitprocessing.parameters.eeli.EELI.compute_parameter",
        operation="eit.eeli",
        parameters={"result_label": result_label},
    )
    eeli_result = _sparse_data_to_array_parameter(
        result, modality="eit", method="eitprocessing.EELI", metadata=dict(metadata)
    )
    session.parameter_results.add(eeli_result)

    _record_step(session, "eit.eeli", metadata=metadata)
    return {"eeli": result, "eeli_result": eeli_result}
