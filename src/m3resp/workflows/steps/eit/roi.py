"""Registered region-of-interest (ROI) EIT workflow steps.

``eit.roi_tiv_lungspace``, ``eit.roi_amplitude_lungspace`` and
``eit.roi_watershed`` find the functional lung space from the breaths, with
the m3resp code in ``m3resp.eit.roi``. ``eit.roi_filter_by_size`` keeps the
groups of connected pixels of a mask that are large enough, and
``eit.geometric_mask`` gives a fixed part of the image. Each step adds its
``PixelMask`` to ``session.pixel_masks``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from m3resp.adapters.eitprocessing_adapter import as_signal, given_breath_events
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, PixelMask, Signal
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.eit.roi import (
    GEOMETRIC_REGIONS,
    amplitude_lungspace,
    filter_mask_by_size,
    tiv_lungspace,
    watershed_lungspace,
)
from m3resp.eit.roi import amplitude as _amplitude
from m3resp.eit.roi import filter_by_size as _filter_by_size
from m3resp.eit.roi import geometric_mask as _geometric_mask
from m3resp.eit.roi import tiv as _tiv
from m3resp.eit.roi import watershed as _watershed
from m3resp.workflows.registry import (
    BREATHS_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _SESSION_ARTIFACT,
    _on_eit_clock,
    _upstream_metadata,
    _use_stored_eit_breaths,
)
from .pixel import _eit_breaths


def _validate_unit_threshold(value: float, *, step: str, param: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{step} '{param}' must be between 0 and 1, got {value!r}.")


def _lungspace_breaths(
    signal: Signal,
    breaths: Any,
    *,
    session: M3Session,
    any_modality: bool,
    action: str,
) -> list[BreathEvent]:
    """Return the global breaths of a lung-space step.

    Args:
        signal (Signal): The impedance waveform of the breaths.
        breaths (Any): The ``breaths`` input, or None.
        session (M3Session): Session with the start times and the stored
            EIT breaths.
        any_modality (bool): True when breaths of every modality are
            accepted (they are moved to the EIT clock); False for EIT
            breaths only.
        action (str): Step name, for the messages.

    Returns:
        list[BreathEvent]: The given breaths, or, when ``breaths`` is None,
            the breaths found on ``signal`` with the default
            ``BreathDetectionSettings``, as in eitprocessing. Found breaths
            that match a breath of ``session.events['eit_breaths']`` are
            that stored breath.

    Raises:
        TypeError: If a given breath is not a ``BreathEvent``.
        ValueError: If ``any_modality`` is False and a given breath has
            another modality than EIT.
    """

    if breaths is None:
        return _use_stored_eit_breaths(
            find_breaths(signal, BreathDetectionSettings()), session
        )
    if any_modality:
        given = given_breath_events(breaths, function=action)
        return _on_eit_clock(given, session, action=action)
    return _eit_breaths(breaths, action=action)


def _add_provenance(
    mask: PixelMask,
    *,
    function: str,
    module: str,
    operation: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Add the provenance of a step to the metadata of its mask, and return it."""

    provenance = _upstream_metadata(
        source_function=function,
        operation=operation,
        parameters=parameters,
        source_package="m3resp",
        implementation=module,
    )
    mask.metadata.update(provenance)
    return provenance


def _breath_origin(breaths: Any) -> str:
    return "timing_data" if breaths is None else "input"


_PIXELS = StepArtifact(
    name="eit_data",
    artifact_type="eit_pixel_signal",
    default_context_key="filtered_eit",
    description="Filtered EIT pixel signal to find the lung space in.",
)

_LUNGSPACE_READS: dict[str, str | None] = {
    "eit_data": "filtered_eit",
    "timing_data": "global_impedance",
    "breaths": "breaths",
    "session": "session",
}


def _timing_artifact(use: str) -> StepArtifact:
    return StepArtifact(
        name="timing_data",
        artifact_type="eit_impedance_waveform",
        default_context_key="global_impedance",
        description=(
            "Impedance waveform (global, or regional from an ROI). Without "
            "'breaths', the step finds the breaths on it with the default "
            f"settings of the breath detection.{use}"
        ),
    )


_ANY_BREATHS = StepArtifact(
    name="breaths",
    artifact_type=BREATHS_ARTIFACT_TYPE,
    required=False,
    description=(
        "Global breaths for the TIV of each pixel; without them, the step "
        "finds the breaths on 'timing_data'. A list of BreathEvents or an "
        "IntervalData of breaths of any modality, for example the breaths "
        "of 'eit.detect_breaths', 'emg.detect_breaths' or "
        "'ventilator.normalize_breaths'. Breaths of another modality are "
        "moved to the EIT clock with the start times of the session."
    ),
)

_EIT_BREATHS = StepArtifact(
    name="breaths",
    artifact_type=BREATHS_ARTIFACT_TYPE,
    required=False,
    description=(
        "Global EIT breaths, for example the breaths of 'eit.detect_breaths'; "
        "without them, the step finds the breaths on 'timing_data'. A list "
        "of BreathEvents or an IntervalData of breaths. The step finds the "
        "breaths of each pixel from them, so breaths of another modality "
        "are refused."
    ),
)

_PHASE_LAG_USE = " The search for the breaths of each pixel also uses it."


def _result_artifact(name: str) -> StepArtifact:
    return StepArtifact(
        name=name,
        artifact_type="pixel_mask",
        description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out.",
        axes=("row", "column"),
    )


def _threshold_parameter(name: str, quantity: str) -> StepParameter:
    return StepParameter(
        name=name,
        value_type="number",
        default=0.15,
        minimum=0.0,
        maximum=1.0,
        description=(
            f"Fraction of the largest mean {quantity} that a pixel must "
            "reach to be included. Must be strictly between 0 and 1."
        ),
    )


@register_step(
    "eit.roi_tiv_lungspace",
    reads=_LUNGSPACE_READS,
    optional_reads=("breaths",),
    writes=("tiv_lungspace_captures", "tiv_lungspace_result"),
    summary="Threshold mean pixel TIV into a functional lung-space mask.",
    description=(
        "Find the functional lung space from the mean tidal impedance "
        "variation (TIV) of each pixel, with m3resp.eit.roi.tiv_lungspace: "
        "the pixels whose mean TIV over the global breaths is at least a "
        "fraction of the largest mean TIV."
    ),
    category="roi",
    modality="eit",
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        _PIXELS,
        _timing_artifact(""),
        _ANY_BREATHS,
        _SESSION_ARTIFACT,
    ),
    parameters=(_threshold_parameter("threshold", "TIV"),),
    output_artifacts=(
        StepArtifact(
            name="tiv_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="'TIV per breath' (IntervalData with one PixelMap per breath) and 'mean TIV' (PixelMap).",
        ),
        _result_artifact("tiv_lungspace_result"),
    ),
    record=True,
)
def roi_tiv_lungspace(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    breaths: Any = None,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Select lung-space pixels by their mean tidal impedance variation (TIV).

    The mask is added to ``session.pixel_masks`` and the processing settings
    are recorded in the session's history.

    Args:
        eit_data (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with time, row and column axes.
        timing_data (Any): Global or regional impedance waveform. Without
            ``breaths``, the breaths are found on it with the default
            ``BreathDetectionSettings``.
        session (M3Session): Session that stores the mask and processing
            history.
        breaths (Any): Global breaths of any modality (a list of
            ``BreathEvent``s or an ``IntervalData`` of breaths), moved to
            the EIT clock with the start times of the session. None finds
            them on ``timing_data``.
        threshold (float): Fraction of the largest mean pixel TIV that a
            pixel must reach, strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``tiv_lungspace_captures`` with ``"TIV per
            breath"`` and ``"mean TIV"``, and ``tiv_lungspace_result``, the
            m3resp ``PixelMask`` (row, column) with 1 for included pixels
            and NaN elsewhere.

    Raises:
        TypeError: If a given breath is not a ``BreathEvent``.
        ValueError: If the threshold is outside the open interval (0, 1),
            there are no breaths, or every TIV value is NaN.
    """

    action = "eit.roi_tiv_lungspace"
    _validate_unit_threshold(threshold, step=action, param="threshold")
    found = _lungspace_breaths(
        as_signal(timing_data),
        breaths,
        session=session,
        any_modality=True,
        action=action,
    )
    captures: dict[str, Any] = {}
    mask = tiv_lungspace(
        as_signal(eit_data), found, threshold=float(threshold), captures=captures
    )
    provenance = _add_provenance(
        mask,
        function=_tiv.METHOD,
        module=_tiv.__name__,
        operation=action,
        parameters={"threshold": threshold, "breaths": _breath_origin(breaths)},
    )
    session.pixel_masks.add(mask)

    return {
        "tiv_lungspace_captures": captures,
        "tiv_lungspace_result": mask,
        "_provenance": provenance,
    }


@register_step(
    "eit.roi_amplitude_lungspace",
    reads=_LUNGSPACE_READS,
    optional_reads=("breaths",),
    writes=("amplitude_lungspace_captures", "amplitude_lungspace_result"),
    summary=(
        "Threshold mean pixel amplitude into a lung-space mask (not "
        "recommended alone; supports eit.roi_watershed)."
    ),
    description=(
        "Find a lung space from the mean amplitude of each pixel, with "
        "m3resp.eit.roi.amplitude_lungspace: the amplitude is the TIV of the "
        "pixel between its own breath start and end of inspiration. Not "
        "recommended as a functional lung space on its own, because it can "
        "include reconstruction artefacts; eit.roi_watershed uses it "
        "together with the TIV."
    ),
    category="roi",
    modality="eit",
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        _PIXELS,
        _timing_artifact(_PHASE_LAG_USE),
        _EIT_BREATHS,
        _SESSION_ARTIFACT,
    ),
    parameters=(_threshold_parameter("threshold", "amplitude"),),
    output_artifacts=(
        StepArtifact(
            name="amplitude_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="'amplitude per breath' (IntervalData with one PixelMap per global breath) and 'mean amplitude' (PixelMap).",
        ),
        _result_artifact("amplitude_lungspace_result"),
    ),
    record=True,
)
def roi_amplitude_lungspace(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    breaths: Any = None,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Threshold mean pixel amplitude into a lung-space mask.

    Warning: this lung space can include reconstruction artefacts, so it is
    not advised on its own. ``eit.roi_watershed`` uses the amplitude
    together with the TIV.

    The amplitude of a pixel in a breath is its TIV between its own breath
    start and end of inspiration, found in the mode "phase shift". The mask
    is added to ``session.pixel_masks`` and the processing settings are
    recorded in the session's history.

    Args:
        eit_data (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with time, row and column axes.
        timing_data (Any): Global or regional impedance waveform, for the
            phase lag of each pixel. Without ``breaths``, the breaths are
            found on it with the default ``BreathDetectionSettings``.
        session (M3Session): Session that stores the mask and processing
            history.
        breaths (Any): Global EIT breaths (a list of ``BreathEvent``s or an
            ``IntervalData`` of breaths) on the time axis of
            ``timing_data``. None finds them on ``timing_data``.
        threshold (float): Fraction of the largest mean pixel amplitude that
            a pixel must reach, strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``amplitude_lungspace_captures`` with
            ``"amplitude per breath"`` and ``"mean amplitude"``, and
            ``amplitude_lungspace_result``, the m3resp ``PixelMask`` (row,
            column) with 1 for included pixels and NaN elsewhere.

    Raises:
        TypeError: If a given breath is not a ``BreathEvent``.
        ValueError: If the threshold is outside the open interval (0, 1),
            a given breath has another modality than EIT, there are no
            breaths, or every amplitude value is NaN.
    """

    action = "eit.roi_amplitude_lungspace"
    _validate_unit_threshold(threshold, step=action, param="threshold")
    signal = as_signal(timing_data)
    found = _lungspace_breaths(
        signal, breaths, session=session, any_modality=False, action=action
    )
    captures: dict[str, Any] = {}
    mask = amplitude_lungspace(
        as_signal(eit_data),
        signal,
        found,
        threshold=float(threshold),
        captures=captures,
    )
    provenance = _add_provenance(
        mask,
        function=_amplitude.METHOD,
        module=_amplitude.__name__,
        operation=action,
        parameters={"threshold": threshold, "breaths": _breath_origin(breaths)},
    )
    session.pixel_masks.add(mask)

    return {
        "amplitude_lungspace_captures": captures,
        "amplitude_lungspace_result": mask,
        "_provenance": provenance,
    }


@register_step(
    "eit.roi_watershed",
    reads=_LUNGSPACE_READS,
    optional_reads=("breaths",),
    writes=("watershed_captures", "watershed_lungspace_result"),
    summary="Derive a lung-space mask with the watershed method (pendelluft-aware).",
    description=(
        "Find the functional lung space with the watershed method of "
        "m3resp.eit.roi.watershed_lungspace: the mean TIV and the mean "
        "amplitude of each pixel over the breaths that have both, watershed "
        "regions around the local maxima of the mean amplitude, and the "
        "regions whose maximum lies inside the TIV mask. Pixels with a low "
        "TIV because of pendelluft can still be included."
    ),
    category="roi",
    modality="eit",
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        _PIXELS,
        _timing_artifact(_PHASE_LAG_USE),
        _EIT_BREATHS,
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold_fraction",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description=(
                "Fraction of the largest mean TIV for the TIV mask; the same "
                "TIV value is the threshold of the amplitude mask. Must be "
                "strictly between 0 and 1."
            ),
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="watershed_captures",
            artifact_type="diagnostic_summary",
            description="The mean maps of the shared breaths, the TIV and amplitude masks, the local maxima and the watershed regions.",
        ),
        _result_artifact("watershed_lungspace_result"),
    ),
    record=True,
)
def roi_watershed(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    breaths: Any = None,
    threshold_fraction: float = 0.15,
) -> dict[str, Any]:
    """Select lung-space pixels with the watershed method.

    The method uses the mean TIV and the mean amplitude of each pixel over
    the breaths that have both (usually all breaths except the first and
    the last). The mask is added to ``session.pixel_masks`` and the
    processing settings are recorded in the session's history.

    Args:
        eit_data (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with time, row and column axes.
        timing_data (Any): Global or regional impedance waveform, for the
            phase lag of each pixel. Without ``breaths``, the breaths are
            found on it with the default ``BreathDetectionSettings``.
        session (M3Session): Session that stores the mask and processing
            history.
        breaths (Any): Global EIT breaths (a list of ``BreathEvent``s or an
            ``IntervalData`` of breaths) on the time axis of
            ``timing_data``. None finds them on ``timing_data``.
        threshold_fraction (float): Fraction of the largest mean pixel TIV
            for the TIV mask, strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``watershed_captures`` with the intermediate
            results (see :func:`m3resp.eit.roi.watershed_lungspace`), and
            ``watershed_lungspace_result``, the m3resp ``PixelMask`` (row,
            column) with 1 for included pixels and NaN elsewhere.

    Raises:
        TypeError: If a given breath is not a ``BreathEvent``.
        ValueError: If the threshold is outside the open interval (0, 1),
            a given breath has another modality than EIT, no breath has
            both TIV and amplitude values, or no local maximum of the mean
            amplitude lies inside the TIV mask.
    """

    action = "eit.roi_watershed"
    _validate_unit_threshold(
        threshold_fraction, step=action, param="threshold_fraction"
    )
    signal = as_signal(timing_data)
    found = _lungspace_breaths(
        signal, breaths, session=session, any_modality=False, action=action
    )
    captures: dict[str, Any] = {}
    mask = watershed_lungspace(
        as_signal(eit_data),
        signal,
        found,
        threshold_fraction=float(threshold_fraction),
        captures=captures,
    )
    provenance = _add_provenance(
        mask,
        function=_watershed.METHOD,
        module=_watershed.__name__,
        operation=action,
        parameters={
            "threshold_fraction": threshold_fraction,
            "breaths": _breath_origin(breaths),
        },
    )
    session.pixel_masks.add(mask)

    return {
        "watershed_captures": captures,
        "watershed_lungspace_result": mask,
        "_provenance": provenance,
    }


def _as_pixel_mask(mask: Any, *, action: str) -> PixelMask:
    """Return an m3resp ``PixelMask`` as it is, or a 2D grid as a ``PixelMask``.

    Args:
        mask (Any): An m3resp ``PixelMask``, or a 2D grid (array, list or
            tuple) with NaN for the pixels outside the region. A 0 in a grid
            becomes NaN, with a warning.
        action (str): Step name, for the message.

    Returns:
        PixelMask: The mask; a grid gets the name ``"mask"``.

    Raises:
        TypeError: If ``mask`` is another type.
        ValueError: If a grid is not 2D.
    """

    if isinstance(mask, PixelMask):
        return mask
    if isinstance(mask, (np.ndarray, list, tuple)):
        return PixelMask(name="mask", values=np.asarray(mask, dtype=float))
    raise TypeError(
        f"{action} needs an m3resp PixelMask or a 2D grid of pixels, for "
        "example 'watershed_lungspace_result'; got "
        f"{type(mask).__name__}."
    )


@register_step(
    "eit.roi_filter_by_size",
    reads={"mask": None, "session": "session"},
    writes=("size_filtered_roi_result",),
    summary="Keep only connected mask regions at or above a minimum size.",
    description=(
        "Keep the groups of connected pixels of a mask that have at least a "
        "minimum number of pixels, with m3resp.eit.roi.filter_mask_by_size. "
        "The result holds 1 for the kept pixels, so a weighted mask "
        "loses its weights."
    ),
    category="roi",
    modality="eit",
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="mask",
            artifact_type="pixel_mask",
            description=(
                "Lung-space mask to filter: an m3resp PixelMask, for example "
                "'watershed_lungspace_result', or a 2D grid with NaN for the "
                "pixels left out."
            ),
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="min_region_size",
            value_type="integer",
            default=10,
            minimum=1,
            unit="pixels",
            description="Smallest number of pixels of a kept group.",
        ),
        StepParameter(
            name="connectivity",
            value_type="integer",
            default=1,
            choices=(1, 2),
            description="Which pixels touch: 1 (pixels that share an edge) or 2 (pixels that share an edge or a corner).",
        ),
    ),
    output_artifacts=(_result_artifact("size_filtered_roi_result"),),
    record=True,
)
def roi_filter_by_size(
    mask: Any,
    *,
    session: M3Session,
    min_region_size: int = 10,
    connectivity: Literal[1, 2] = 1,
) -> dict[str, Any]:
    """Remove the groups of connected pixels that are smaller than a minimum.

    The filtered mask, named ``"size_filtered_roi_mask"``, is added to
    ``session.pixel_masks`` and the processing settings are recorded in the
    session's history.

    Args:
        mask (Any): An m3resp ``PixelMask``, or a 2D grid with NaN for the
            pixels left out.
        session (M3Session): Session that stores the filtered mask and
            processing history.
        min_region_size (int): Smallest number of connected pixels of a
            kept group; positive.
        connectivity (Literal[1, 2]): ``1`` joins pixels that share an edge;
            ``2`` also joins pixels that share a corner.

    Returns:
        dict[str, Any]: ``size_filtered_roi_result``, the m3resp
            ``PixelMask`` (row, column) with 1 for the pixels of the kept
            groups and NaN elsewhere.

    Raises:
        TypeError: If ``mask`` is not a ``PixelMask`` or a grid.
        ValueError: If ``min_region_size`` is zero or negative, a grid is
            not 2D, or ``connectivity`` is not 1 or 2.
        RuntimeError: If no group has ``min_region_size`` pixels or more.
    """

    action = "eit.roi_filter_by_size"
    if min_region_size <= 0:
        raise ValueError(
            f"{action} 'min_region_size' must be positive, got {min_region_size!r}."
        )
    result = filter_mask_by_size(
        _as_pixel_mask(mask, action=action),
        min_region_size=min_region_size,
        connectivity=connectivity,
        name="size_filtered_roi_mask",
    )
    provenance = _add_provenance(
        result,
        function=_filter_by_size.METHOD,
        module=_filter_by_size.__name__,
        operation=action,
        parameters={
            "min_region_size": min_region_size,
            "connectivity": connectivity,
        },
    )
    session.pixel_masks.add(result)

    return {"size_filtered_roi_result": result, "_provenance": provenance}


@register_step(
    "eit.geometric_mask",
    reads={"eit_data": "filtered_eit", "session": "session"},
    writes=("geometric_mask_result",),
    summary="Make a mask of a fixed part of the EIT image (half, layer or quadrant).",
    description=(
        "Make a mask of a fixed part of the EIT image: the ventral or dorsal "
        "half, the anatomical right or left half, one of four layers from "
        "ventral to dorsal, or one of four quadrants. The anatomical right "
        "of the patient is the left side of the image."
    ),
    category="roi",
    modality="eit",
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="EIT pixel signal; it gives the number of rows and columns of the mask.",
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="region",
            value_type="choice",
            required=True,
            choices=GEOMETRIC_REGIONS,
            description=(
                "Part of the image. 'ventral' and 'dorsal' are the top and "
                "bottom halves of the rows; 'anatomical right' and "
                "'anatomical left' are the left and right halves of the "
                "columns; 'layer 1' to 'layer 4' are quarters of the rows "
                "from ventral to dorsal; 'quadrant 1' to 'quadrant 4' are "
                "ventral right, ventral left, dorsal right and dorsal left."
            ),
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="geometric_mask_result",
            artifact_type="pixel_mask",
            description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out. Its name is the region.",
            axes=("row", "column"),
        ),
    ),
    record=True,
)
def geometric_mask(
    *,
    eit_data: Any,
    session: M3Session,
    region: str,
) -> dict[str, Any]:
    """Make the mask of a fixed part of the EIT image.

    The mask is added to ``session.pixel_masks`` and the settings are
    recorded in the session's history.

    Args:
        eit_data: EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with time, row and column axes. Only the number of
            rows and columns is used.
        session: Session that stores the mask and processing history.
        region: One of the 12 region names of
            :func:`m3resp.eit.roi.geometric_mask`.

    Returns:
        dict[str, Any]: ``geometric_mask_result``, the m3resp ``PixelMask``
            with 1 for the pixels of the region and NaN elsewhere.

    Raises:
        ValueError: If ``region`` is unknown, the images are not (time, row,
            column), or their rows or columns do not divide into the parts
            of the region.
    """

    pixels = as_signal(eit_data)
    values = np.asarray(pixels.values)
    if values.ndim != 3:
        raise ValueError(
            "eit.geometric_mask needs EIT images with time, row and column "
            f"axes, got {values.ndim} axes."
        )
    mask = _geometric_mask(region, shape=(values.shape[1], values.shape[2]))
    metadata = _upstream_metadata(
        source_function="m3resp.eit.roi.geometric_mask",
        operation="eit.geometric_mask",
        parameters={"region": region},
        source_package="m3resp",
        implementation="m3resp.eit.roi.geometric",
    )
    mask.metadata.update(metadata)
    session.pixel_masks.add(mask)

    return {"geometric_mask_result": mask, "_provenance": metadata}
