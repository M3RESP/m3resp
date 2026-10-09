"""Registered EIT steps on impedance waveforms (global, regional and functional impedance, breaths, TIV, EELI)."""

from __future__ import annotations

from typing import Any

from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
    continuous_data_to_signal,
    sparse_data_to_interval_data,
)
from m3resp.core.session import M3Session
from m3resp.data import PixelMask
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _EITPROCESSING,
    _SESSION_ARTIFACT,
    _breaths_used_by,
    _record_step,
    _upstream_metadata,
)


@register_step(
    "eit.global_impedance",
    reads={"signal": "filtered_eit", "eit_sequence": "eit_sequence"},
    writes=("global_impedance",),
    summary="Compute and store the global impedance (sum of all pixels) of an EIT signal.",
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
            description="Sequence the global impedance is added onto.",
            public=False,
            compatibility_only=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="global_impedance",
            artifact_type="eit_impedance_waveform",
            description="Global impedance: the sum of all pixels, with no mask.",
            compatibility_only=True,
        ),
    ),
)
def global_impedance(
    signal: Any,
    *,
    eit_sequence: Any,
) -> dict[str, Any]:
    """Sum pixel impedance into a waveform and add it to the EIT sequence.

    Args:
        signal: eitprocessing pixel data with time, row and column axes.
        eit_sequence: Sequence whose continuous-data collection receives
            the summed impedance waveform.

    Returns:
        dict[str, Any]: ``global_impedance`` containing the upstream waveform
            with its time axis and impedance unit.
    """

    summed = signal.get_summed_impedance()
    add_to_collection(eit_sequence.continuous_data, summed)
    return {"global_impedance": summed}


def _mask_label(mask: Any) -> str | None:
    """Return the name of an m3resp mask or the label of an upstream mask."""

    if isinstance(mask, PixelMask):
        return mask.name
    label = getattr(mask, "label", None)
    return label if isinstance(label, str) else None


def _impedance_in_mask(
    signal: Any,
    mask: Any,
    *,
    eit_sequence: Any,
    session: M3Session,
    step: str,
    output: str,
) -> dict[str, Any]:
    """Sum pixel impedance inside a mask, store the waveform and record the step.

    The upstream waveform goes onto the EIT sequence; a native ``Signal``
    with channel ``output`` goes into ``session.signals``.
    """

    waveform = session.eit_adapter.get_regional_impedance(signal, mask, label=output)
    add_to_collection(eit_sequence.continuous_data, waveform)
    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.roi.PixelMask.apply + "
            "eitprocessing.datahandling.eitdata.EITData.get_summed_impedance"
        ),
        operation=step,
        parameters={"mask": _mask_label(mask)},
    )
    native = continuous_data_to_signal(
        waveform,
        modality="eit",
        channel=output,
        processing_state="intermediate",
        source="eitprocessing",
        method="eitprocessing.PixelMask.apply",
        name=output,
    )
    native.metadata.update(metadata)
    session.signals.add(native)
    _record_step(session, step, metadata=metadata)
    return {output: waveform, f"{output}_signal": native}


_PIXEL_SIGNAL_TO_SUM = StepArtifact(
    name="signal",
    artifact_type="eit_pixel_signal",
    default_context_key="filtered_eit",
    description="Filtered EIT pixel signal to sum.",
    compatibility_only=True,
)

_SEQUENCE_FOR_WAVEFORM = StepArtifact(
    name="eit_sequence",
    artifact_type="eit_sequence",
    description="Sequence the waveform is added onto.",
    public=False,
    compatibility_only=True,
)


@register_step(
    "eit.regional_impedance",
    reads={
        "signal": "filtered_eit",
        "mask": None,
        "eit_sequence": "eit_sequence",
        "session": "session",
    },
    writes=("regional_impedance", "regional_impedance_signal"),
    summary="Sum pixel impedance inside a mask into a regional impedance waveform.",
    description="Multiply each pixel by its mask value and sum the pixels per frame, giving the regional impedance waveform of any region (a layer, a quadrant, the lung space).",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.signals",),
    parameters_reviewed=True,
    input_artifacts=(
        _PIXEL_SIGNAL_TO_SUM,
        StepArtifact(
            name="mask",
            artifact_type=ANY_ARTIFACT_TYPE,
            description=(
                "Mask of the region: an m3resp PixelMask (e.g. "
                "'watershed_lungspace_result'), an eitprocessing PixelMask, "
                "or a 2D grid with NaN for pixels outside the region."
            ),
        ),
        _SEQUENCE_FOR_WAVEFORM,
        _SESSION_ARTIFACT,
    ),
    output_artifacts=(
        StepArtifact(
            name="regional_impedance",
            artifact_type="eit_impedance_waveform",
            description="Regional impedance: the sum of the pixels inside the mask.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="regional_impedance_signal",
            artifact_type="signal",
            description="Native Signal wrapping the regional impedance.",
        ),
    ),
)
def regional_impedance(
    signal: Any,
    *,
    mask: Any,
    eit_sequence: Any,
    session: M3Session,
) -> dict[str, Any]:
    """Sum pixel impedance inside a mask into one waveform.

    Each pixel is multiplied by its mask value (NaN leaves it out, a weight
    counts it partly) and the pixels are summed per frame. The waveform is
    added to the EIT sequence and to ``session.signals``, and the step is
    recorded in the session's history.

    Args:
        signal: eitprocessing pixel data with time, row and column axes.
        mask: An m3resp or eitprocessing ``PixelMask``, or a 2D grid with
            NaN for pixels outside the region; its shape must match the image.
        eit_sequence: Sequence whose continuous-data collection receives the
            waveform.
        session: Session that stores the waveform and processing history.

    Returns:
        dict[str, Any]: ``regional_impedance``, the upstream waveform with its
            time axis, in arbitrary units (AU), and ``regional_impedance_signal``, the
            same waveform as a native ``Signal`` (also added to
            ``session.signals``).
    """

    return _impedance_in_mask(
        signal,
        mask,
        eit_sequence=eit_sequence,
        session=session,
        step="eit.regional_impedance",
        output="regional_impedance",
    )


@register_step(
    "eit.functional_impedance",
    reads={
        "signal": "filtered_eit",
        "mask": "tiv_lungspace_result",
        "eit_sequence": "eit_sequence",
        "session": "session",
    },
    writes=("functional_impedance", "functional_impedance_signal"),
    summary="Sum pixel impedance inside the functional lung space (functional impedance).",
    description="Regional impedance of the functional (ventilated) lung space: each pixel is multiplied by its lung-space mask value and the pixels are summed per frame.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.signals",),
    parameters_reviewed=True,
    input_artifacts=(
        _PIXEL_SIGNAL_TO_SUM,
        StepArtifact(
            name="mask",
            artifact_type=ANY_ARTIFACT_TYPE,
            default_context_key="tiv_lungspace_result",
            description=(
                "Functional lung-space mask, e.g. from eit.roi_tiv_lungspace, "
                "eit.roi_amplitude_lungspace or eit.roi_watershed."
            ),
        ),
        _SEQUENCE_FOR_WAVEFORM,
        _SESSION_ARTIFACT,
    ),
    output_artifacts=(
        StepArtifact(
            name="functional_impedance",
            artifact_type="eit_impedance_waveform",
            description="Functional impedance: the sum of the pixels inside the functional lung space.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="functional_impedance_signal",
            artifact_type="signal",
            description="Native Signal wrapping the functional impedance.",
        ),
    ),
)
def functional_impedance(
    signal: Any,
    *,
    mask: Any,
    eit_sequence: Any,
    session: M3Session,
) -> dict[str, Any]:
    """Sum pixel impedance inside the functional lung space into one waveform.

    This is the regional impedance waveform of the functional (ventilated)
    lung space, commonly called the functional impedance. The calculation is
    the same as ``eit.regional_impedance``.

    Args:
        signal: eitprocessing pixel data with time, row and column axes.
        mask: Functional lung-space mask (m3resp or eitprocessing
            ``PixelMask``, or a 2D grid with NaN outside the lung space).
        eit_sequence: Sequence whose continuous-data collection receives the
            waveform.
        session: Session that stores the waveform and processing history.

    Returns:
        dict[str, Any]: ``functional_impedance``, the upstream waveform with its
            time axis, in arbitrary units (AU), and ``functional_impedance_signal``, the
            same waveform as a native ``Signal`` (also added to
            ``session.signals``).
    """

    return _impedance_in_mask(
        signal,
        mask,
        eit_sequence=eit_sequence,
        session=session,
        step="eit.functional_impedance",
        output="functional_impedance",
    )


@register_step(
    "eit.detect_breaths",
    reads={"signal": "detection_signal"},
    writes=("breath_intervals", "breath_detector"),
    summary="Detect breaths on a continuous EIT impedance signal.",
    description="Detect breath start/end times on an impedance waveform (global, or regional from an ROI) via eitprocessing's BreathDetection.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_impedance_waveform",
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
    """Detect breaths on an impedance waveform with eitprocessing.

    Args:
        signal: Global or regional impedance waveform with times in seconds.
        min_duration_s: Minimum separation used by the breath detector,
            in seconds.

    Returns:
        dict[str, Any]: ``breath_intervals`` as upstream detected breaths and
            ``breath_detector`` as the configured detector for later steps.
            Breath times retain the waveform's time axis.
    """

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
        "session": "session",
    },
    writes=("continuous_tiv", "continuous_tiv_result"),
    summary="Compute continuous tidal impedance variation (TIV).",
    description="Compute per-breath tidal impedance variation on an impedance waveform (global, or regional from an ROI) via eitprocessing's TIV.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    parameters_reviewed=True,
    session_writes=("session.interval_data",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) to compute TIV on.",
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
    output_artifacts=(
        StepArtifact(
            name="continuous_tiv",
            artifact_type="eit_sparse_data",
            description="Per-breath continuous TIV values (upstream SparseData).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="continuous_tiv_result",
            artifact_type="interval_data",
            description="One TIV value per breath, each stored with its breath (IntervalData).",
        ),
    ),
)
def continuous_tiv(
    signal: Any,
    *,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
) -> dict[str, Any]:
    """Compute tidal impedance variation (TIV) for each detected breath.

    The upstream result is added to the sequence's sparse-data collection.
    Values paired with their breaths are added to ``session.interval_data``,
    and the processing step is recorded in the session's history.

    Args:
        signal: Global or regional impedance waveform to measure.
        eit_sequence: Sequence that stores the upstream result.
        breath_detector: Detector used to identify breaths on ``signal``.
        session: Session that stores the per-breath values and history.

    Returns:
        dict[str, Any]: ``continuous_tiv`` as upstream sparse data and
            ``continuous_tiv_result`` as ``IntervalData`` with one value per
            breath, in detector order and in the signal's impedance unit.

    Raises:
        ValueError: If the numbers of values and detected breaths differ.
    """

    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(breath_detection=breath_detector).compute_parameter(
        signal, sequence=eit_sequence, store=False, result_label="continuous_tivs"
    )
    add_to_collection(eit_sequence.sparse_data, result)

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.parameters.tidal_impedance_variation.TIV.compute_parameter"
        ),
        operation="eit.continuous_tiv",
        parameters={"result_label": "continuous_tivs"},
    )
    continuous_tiv_result = sparse_data_to_interval_data(
        result,
        _breaths_used_by(breath_detector, signal, session),
        modality="eit",
        method="eitprocessing.TIV",
        metadata=dict(metadata),
    )
    session.interval_data.add(continuous_tiv_result)

    _record_step(session, "eit.continuous_tiv", metadata=metadata)
    return {"continuous_tiv": result, "continuous_tiv_result": continuous_tiv_result}


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
    session_writes=("session.interval_data",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) to compute EELI on.",
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
            artifact_type="interval_data",
            description="One EELI value per breath, each stored with its breath (IntervalData).",
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
    """Compute end-expiratory lung impedance (EELI) for each detected breath.

    The upstream result is stored in the EIT sequence. Values paired with
    their breaths are added to ``session.interval_data``, and the processing
    step is recorded in the session's history.

    Args:
        signal: Global or regional impedance waveform to measure.
        eit_sequence: Sequence that stores the upstream result.
        breath_detector: Detector used to identify breaths on ``signal``.
        session: Session that stores the per-breath values and history.
        result_label: Name used for the upstream and per-breath results.

    Returns:
        dict[str, Any]: ``eeli`` as upstream sparse data and ``eeli_result``
            as ``IntervalData`` with one value per breath, in detector order
            and in the signal's impedance unit.

    Raises:
        ValueError: If the numbers of values and detected breaths differ.
    """

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
    eeli_result = sparse_data_to_interval_data(
        result,
        _breaths_used_by(breath_detector, signal, session),
        modality="eit",
        method="eitprocessing.EELI",
        metadata=dict(metadata),
    )
    session.interval_data.add(eeli_result)

    _record_step(session, "eit.eeli", metadata=metadata)
    return {"eeli": result, "eeli_result": eeli_result}
