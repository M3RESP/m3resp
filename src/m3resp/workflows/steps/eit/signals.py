"""Registered EIT steps on impedance waveforms (global, regional and functional impedance, breaths, TIV, EELI)."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    EITPROCESSING_FIELDS,
    add_to_collection,
    as_breath_detection,
    as_continuous_data,
    as_signal,
    breaths_to_interval_data,
    eit_data_to_signal,
    eitprocessing_fields,
    signal_to_continuous_data,
    sparse_data_to_interval_data,
)
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.core.session import M3Session
from m3resp.data import PixelMask, Signal
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.eit.processing import global_impedance as summed_pixels
from m3resp.eit.processing import regional_impedance as regional_impedance_of
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
    _upstream_metadata,
)


@register_step(
    "eit.global_impedance",
    reads={"signal": "filtered_eit", "eit_sequence": "eit_sequence"},
    writes=("global_impedance",),
    summary="Compute and store the global impedance (sum of all pixels) of an EIT signal.",
    description="Sum the pixel impedance of each image into the global impedance waveform used for breath detection. m3resp computes the sum (a port of eitprocessing's EITData.get_summed_impedance).",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT images (pixel impedance) to sum.",
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
            description="Global impedance: the sum of all pixels, with no mask, as a Signal.",
        ),
    ),
)
def global_impedance(
    signal: Any,
    *,
    eit_sequence: Any,
) -> dict[str, Any]:
    """Sum the pixels of each image into a waveform and add it to the EIT sequence.

    The sum is ``m3resp.eit.processing.global_impedance``. The waveform is
    named ``"summed <label>"``, where ``<label>`` is the eitprocessing label of
    the images, as eitprocessing names it.

    Args:
        signal: EIT images as a ``Signal`` (or an eitprocessing ``EITData``),
            with time, row and column axes.
        eit_sequence: Sequence whose continuous-data collection receives
            the waveform as an eitprocessing ``ContinuousData``.

    Returns:
        dict[str, Any]: ``global_impedance``, the waveform as a ``Signal``
            with the time axis of ``signal``, in arbitrary units (AU). Its
            metadata names the function that computed it.
    """

    pixels = signal if isinstance(signal, Signal) else eit_data_to_signal(signal)
    label = eitprocessing_fields(pixels).get("label", pixels.name)
    summed = summed_pixels(pixels, name=f"summed {label}")
    # eitprocessing writes the unit as "AU"; the Signal holds m3resp's "a.u.".
    summed.metadata[EITPROCESSING_FIELDS] = {"type": "ContinuousData", "unit": "AU"}
    summed.metadata.update(
        _upstream_metadata(
            source_function="m3resp.eit.processing.global_impedance",
            operation="eit.global_impedance",
            parameters={},
            source_package="m3resp",
            implementation="m3resp.eit.processing.impedance",
        )
    )
    add_to_collection(eit_sequence.continuous_data, signal_to_continuous_data(summed))
    return {"global_impedance": summed}


def _mask_label(mask: Any) -> str | None:
    """Return the name of an m3resp mask or the label of an upstream mask."""

    if isinstance(mask, PixelMask):
        return mask.name
    label = getattr(mask, "label", None)
    return label if isinstance(label, str) else None


def _as_mask(mask: Any) -> PixelMask:
    """Return ``mask`` as an m3resp ``PixelMask``.

    An eitprocessing ``PixelMask`` keeps its values as they are, zeros and
    weights included, because eitprocessing has checked them. A plain grid
    runs the checks of ``PixelMask``, so a 0 becomes NaN, as in
    eitprocessing.

    Args:
        mask (Any): An m3resp or eitprocessing ``PixelMask``, or a 2D grid
            of numbers with NaN for pixels outside the region.

    Returns:
        PixelMask: The mask.

    Raises:
        UnsupportedWorkflowError: If ``mask`` is of another type or is not a
            2D grid of numbers.
    """

    if isinstance(mask, PixelMask):
        return mask
    upstream = hasattr(mask, "mask")
    value = mask.mask if upstream else mask
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise UnsupportedWorkflowError(
            "An ROI mask must be an m3resp PixelMask, an eitprocessing "
            f"PixelMask, or a 2D array of pixels; got {type(mask).__name__}."
        )
    try:
        grid = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise UnsupportedWorkflowError(
            "An ROI mask must be a 2D (row, column) array of numbers."
        ) from error
    if grid.ndim != 2:
        raise UnsupportedWorkflowError(
            "An ROI mask must be a 2D (row, column) array of pixels; got "
            f"shape {grid.shape}."
        )
    try:
        if upstream:
            return PixelMask(
                name=_mask_label(mask) or "mask",
                values=grid,
                keep_zeros=True,
                suppress_value_range_error=True,
                suppress_all_nan_warning=True,
            )
        return PixelMask(name="mask", values=grid)
    except TypeError as error:
        raise UnsupportedWorkflowError(
            "An ROI mask must be a 2D (row, column) array of numbers "
            "that can be stored as floats without changing their values."
        ) from error


def _impedance_in_mask(
    signal: Any,
    mask: Any,
    *,
    eit_sequence: Any,
    session: M3Session,
    step: str,
    output: str,
) -> dict[str, Any]:
    """Sum pixel impedance inside a mask and store the waveform.

    ``m3resp.eit.processing.regional_impedance`` computes the sum. The
    waveform goes into ``session.signals`` as a ``Signal`` with channel
    ``output``, and onto the EIT sequence as an eitprocessing
    ``ContinuousData`` labelled ``output``. The returned ``"_provenance"``
    details go into the calling step's provenance record.
    """

    pixels = signal if isinstance(signal, Signal) else eit_data_to_signal(signal)
    native = regional_impedance_of(pixels, _as_mask(mask), name=output, channel=output)
    native.metadata[EITPROCESSING_FIELDS] = {
        "type": "ContinuousData",
        "label": output,
        "name": output.replace("_", " ").capitalize(),
        "description": "Impedance summed over the pixels of a mask.",
        "unit": "AU",
    }
    metadata = _upstream_metadata(
        source_function="m3resp.eit.processing.regional_impedance",
        operation=step,
        parameters={"mask": _mask_label(mask)},
        source_package="m3resp",
        implementation="m3resp.eit.processing.impedance",
    )
    native.metadata.update(metadata)
    add_to_collection(eit_sequence.continuous_data, signal_to_continuous_data(native))
    session.signals.add(native)
    return {output: native, f"{output}_signal": native, "_provenance": metadata}


_PIXEL_SIGNAL_TO_SUM = StepArtifact(
    name="signal",
    artifact_type="eit_pixel_signal",
    default_context_key="filtered_eit",
    description="Filtered EIT images (pixel impedance) to sum.",
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
            description="Regional impedance: the sum of the pixels inside the mask, as a Signal.",
        ),
        StepArtifact(
            name="regional_impedance_signal",
            artifact_type="eit_impedance_waveform",
            description="The same Signal as 'regional_impedance'.",
        ),
    ),
    record=True,
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
        signal: EIT images as a ``Signal`` (or an eitprocessing ``EITData``),
            with time, row and column axes.
        mask: An m3resp or eitprocessing ``PixelMask``, or a 2D grid with
            NaN for pixels outside the region; its shape must match the image.
        eit_sequence: Sequence whose continuous-data collection receives the
            waveform.
        session: Session that stores the waveform and processing history.

    Returns:
        dict[str, Any]: ``regional_impedance`` and
            ``regional_impedance_signal``, the same waveform as a ``Signal``
            with the time axis of ``signal``, in arbitrary units (AU), also
            added to ``session.signals``.
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
            description="Functional impedance: the sum of the pixels inside the functional lung space, as a Signal.",
        ),
        StepArtifact(
            name="functional_impedance_signal",
            artifact_type="eit_impedance_waveform",
            description="The same Signal as 'functional_impedance'.",
        ),
    ),
    record=True,
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
        signal: EIT images as a ``Signal`` (or an eitprocessing ``EITData``),
            with time, row and column axes.
        mask: Functional lung-space mask (m3resp or eitprocessing
            ``PixelMask``, or a 2D grid with NaN outside the lung space).
        eit_sequence: Sequence whose continuous-data collection receives the
            waveform.
        session: Session that stores the waveform and processing history.

    Returns:
        dict[str, Any]: ``functional_impedance`` and
            ``functional_impedance_signal``, the same waveform as a ``Signal``
            with the time axis of ``signal``, in arbitrary units (AU), also
            added to ``session.signals``.
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
    description="Detect the breaths (start, peak, end) of an impedance waveform (global, or regional from an ROI) from its peaks and valleys (m3resp.eit.processing.find_breaths).",
    category="detection",
    modality="eit",
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_impedance_waveform",
            default_context_key="detection_signal",
            description="Impedance waveform (typically the detection-window slice) to detect breaths on.",
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
            artifact_type="interval_data",
            description="The detected breaths as BreathEvents (IntervalData without values).",
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="The settings of the breath detection (BreathDetectionSettings), for the TIV, EELI and pixel steps.",
            public=False,
        ),
    ),
)
def detect_breaths(signal: Any, *, min_duration_s: float = 2 / 3) -> dict[str, Any]:
    """Detect the breaths of an impedance waveform.

    Args:
        signal: Global or regional impedance waveform as a ``Signal`` (or an
            eitprocessing ``ContinuousData``), times in seconds.
        min_duration_s: Shortest expected breath, in seconds.

    Returns:
        dict[str, Any]: ``breath_intervals``, an ``IntervalData`` whose
            intervals are the breaths as ``BreathEvent``s (start, peak in
            ``extremum_time``, end) on the time axis of the waveform, and
            ``breath_detector``, the ``BreathDetectionSettings`` for later
            steps.
    """

    settings = BreathDetectionSettings(minimum_duration=min_duration_s)
    return {
        "breath_intervals": breaths_to_interval_data(
            find_breaths(as_signal(signal), settings)
        ),
        "breath_detector": settings,
    }


@register_step(
    "eit.normalize_breaths",
    reads={"breath_intervals": "breath_intervals", "session": "session"},
    writes=(),
    summary="Store detected EIT breaths as session events.",
    description="Store the breaths from 'eit.detect_breaths' as BreathEvents on the session under 'eit_breaths'.",
    category="detection",
    modality="eit",
    parameters_reviewed=True,
    session_writes=("session.events.eit_breaths",),
    input_artifacts=(
        StepArtifact(
            name="breath_intervals",
            artifact_type="interval_data",
            default_context_key="breath_intervals",
            description="Breaths from 'eit.detect_breaths' to store.",
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
            description="Breath detection settings from 'eit.detect_breaths'.",
            public=False,
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
    record=True,
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
        signal: Global or regional impedance waveform to measure, as a
            ``Signal`` (or an eitprocessing ``ContinuousData``).
        eit_sequence: Sequence that stores the upstream result.
        breath_detector: Settings of the breath detection on ``signal``
            (``BreathDetectionSettings``), or an eitprocessing
            ``BreathDetection``.
        session: Session that stores the per-breath values and history.

    Returns:
        dict[str, Any]: ``continuous_tiv`` as upstream sparse data and
            ``continuous_tiv_result`` as ``IntervalData`` with one value per
            breath, in detector order and in the signal's impedance unit.

    Raises:
        ValueError: If the numbers of values and detected breaths differ.
    """

    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(
        breath_detection=as_breath_detection(breath_detector)
    ).compute_parameter(
        as_continuous_data(signal),
        sequence=eit_sequence,
        store=False,
        result_label="continuous_tivs",
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

    return {
        "continuous_tiv": result,
        "continuous_tiv_result": continuous_tiv_result,
        "_provenance": metadata,
    }


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
            description="Breath detection settings from 'eit.detect_breaths'.",
            public=False,
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
    record=True,
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
        signal: Global or regional impedance waveform to measure, as a
            ``Signal`` (or an eitprocessing ``ContinuousData``).
        eit_sequence: Sequence that stores the upstream result.
        breath_detector: Settings of the breath detection on ``signal``
            (``BreathDetectionSettings``), or an eitprocessing
            ``BreathDetection``.
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

    return {"eeli": result, "eeli_result": eeli_result, "_provenance": metadata}
