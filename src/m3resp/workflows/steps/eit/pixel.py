"""Registered pixel-level EIT workflow steps.

``eit.pixel_breaths`` finds the breaths of each pixel. ``eit.pixel_tiv`` and
``eit.pixel_eeli`` give the TIV and the EELI of each pixel, with the timing
of the global breaths or of the breaths of each pixel. The calculations are
the m3resp code in ``m3resp.eit.processing``. Each step stores an
``IntervalData`` on ``session.interval_data``: per-pixel TIV and EELI as one
``PixelMap`` per breath, and per-pixel breath timing as one (row, column, 3)
grid of times per breath.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    as_signal,
    given_breath_events,
    pixel_sign_breaths,
)
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, IntervalData, PixelBreaths, Signal
from m3resp.data.pixel_breaths import PIXEL_BREATH_POINTS
from m3resp.eit.processing import (
    TIV_METHODS,
    BreathDetectionSettings,
    find_breaths,
    find_pixel_breaths,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
)
from m3resp.eit.processing.eeli import PIXEL_METHOD as PIXEL_EELI_FUNCTION
from m3resp.eit.processing.pixel_breath import METHOD as PIXEL_BREATH_FUNCTION
from m3resp.eit.processing.pixel_breath import (
    PHASE_CORRECTION_MODES,
    PhaseCorrectionMode,
)
from m3resp.eit.processing.tidal_impedance_variation import (
    PIXEL_METHOD as PIXEL_TIV_FUNCTION,
)
from m3resp.modalities.names import normalize_modality
from m3resp.workflows.registry import (
    BREATHS_ARTIFACT_TYPE,
    PIXEL_BREATHS_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _SESSION_ARTIFACT,
    _breaths_used_by,
    _per_breath_values,
    _upstream_metadata,
    _use_stored_eit_breaths,
)

#: Timing of the pixel TIV and the pixel EELI: the global breaths
#: ("continuous") or the breaths of each pixel ("pixel").
PixelTiming = Literal["pixel", "continuous"]
_PIXEL_TIMINGS = ("pixel", "continuous")


_PIXEL_SIGNAL = StepArtifact(
    name="eit_data",
    artifact_type="eit_pixel_signal",
    default_context_key="filtered_eit",
    description=(
        "EIT pixel signal to compute the values on. Defaults to the "
        "filtered signal, but any pixel signal can be bound here."
    ),
)

_TIMING_WAVEFORM = StepArtifact(
    name="signal",
    artifact_type="eit_impedance_waveform",
    default_context_key="global_impedance",
    description="Impedance waveform (global, or regional from an ROI) supplying breath timing.",
)

_BREATH_DETECTOR = StepArtifact(
    name="breath_detector",
    artifact_type="eit_breath_detector",
    required=False,
    description="Breath detection settings from 'eit.detect_breaths', used when neither 'breaths' nor 'pixel_breaths' is connected.",
    public=False,
)

_GLOBAL_BREATHS = StepArtifact(
    name="breaths",
    artifact_type=BREATHS_ARTIFACT_TYPE,
    required=False,
    description=(
        "Global breaths; without them, the step finds the breaths with "
        "'breath_detector'. A list of BreathEvents or an IntervalData of "
        "breaths. With the 'continuous' timing, breaths of any modality, for "
        "example the breaths of 'eit.detect_breaths', 'emg.detect_breaths' or "
        "'ventilator.normalize_breaths'; breaths of another modality are moved "
        "to the EIT clock with the start times of the session. With the "
        "'pixel' timing and no 'pixel_breaths', EIT breaths only: the step "
        "finds the pixel breaths from them."
    ),
)

_PIXEL_BREATHS = StepArtifact(
    name="pixel_breaths",
    artifact_type=PIXEL_BREATHS_ARTIFACT_TYPE,
    default_context_key="pixel_breaths",
    required=False,
    description=(
        "The breaths of each pixel from 'eit.pixel_breaths' (PixelBreaths), "
        "used by the 'pixel' timing only. Without them, the 'pixel' timing "
        "finds the pixel breaths in the mode 'negative amplitude'."
    ),
)


def _timing_parameter(name: str, quantity: str) -> StepParameter:
    """Return the step parameter that chooses the global or the pixel breath timing.

    Args:
        name (str): Name of the parameter, ``"tiv_timing"`` or
            ``"eeli_timing"``.
        quantity (str): ``"TIV"`` or ``"EELI"``, for the description.

    Returns:
        StepParameter: A choice of ``"continuous"`` (default) or ``"pixel"``.
    """

    return StepParameter(
        name=name,
        value_type="choice",
        default="continuous",
        choices=_PIXEL_TIMINGS,
        description=(
            f"'continuous': the {quantity} of each pixel at the times of the "
            "global breaths of the waveform bound to 'signal' (the global "
            "impedance by default, or a regional impedance from an ROI). "
            f"'pixel': the {quantity} of each pixel at the times of its own "
            "breaths."
        ),
    )


def _eit_breaths(breaths: Any, *, action: str) -> list[BreathEvent]:
    """Return given global breaths for the pixel breath search, EIT breaths only.

    Args:
        breaths (Any): A list of ``BreathEvent``s, an m3resp ``IntervalData``
            of them, or an eitprocessing ``IntervalData`` of ``Breath``s.
        action (str): Step name, for the message.

    Returns:
        list[BreathEvent]: The breaths in input order.

    Raises:
        TypeError: If an item of a list or of an m3resp ``IntervalData`` is
            not a ``BreathEvent``.
        ValueError: If a breath has another modality than EIT.
    """

    given = given_breath_events(breaths, function=action)
    others = sorted({normalize_modality(breath.modality) for breath in given} - {"eit"})
    if others:
        raise ValueError(
            f"{action} finds the pixel breaths from EIT breaths only; got "
            f"breaths of {', '.join(others)}. The search windows of the pixel "
            "breaths lie between the ends of inspiration of the impedance "
            "breaths."
        )
    return given


def _pixel_breath_timing(
    pixels: Signal,
    signal: Any,
    *,
    pixel_breaths: Any,
    breaths: Any,
    breath_detector: Any,
    session: M3Session,
    action: str,
) -> PixelBreaths:
    """Return the breaths of each pixel for the 'pixel' timing.

    Args:
        pixels (Signal): The EIT images.
        signal (Any): The impedance waveform of the global breaths.
        pixel_breaths (Any): The ``PixelBreaths`` of ``eit.pixel_breaths``,
            or None to find them here.
        breaths (Any): Given EIT breaths, or None.
        breath_detector (Any): Breath detection settings, used when both
            ``pixel_breaths`` and ``breaths`` are None.
        session (M3Session): Session with the stored EIT breaths.
        action (str): Step name, for the messages.

    Returns:
        PixelBreaths: ``pixel_breaths`` as it is, or the pixel breaths found
            in the mode "negative amplitude". For breaths found with
            ``breath_detector``, the sign of each pixel comes from the
            breaths found with the default settings, as in eitprocessing.

    Raises:
        TypeError: If ``pixel_breaths`` is not a ``PixelBreaths``.
        ValueError: If no breaths are given and ``breath_detector`` is None,
            or the given breaths are not EIT breaths.
    """

    if pixel_breaths is not None:
        if not isinstance(pixel_breaths, PixelBreaths):
            raise TypeError(
                f"{action}: 'pixel_breaths' must be the PixelBreaths of "
                f"'eit.pixel_breaths'; got {type(pixel_breaths).__name__}. "
                "PixelBreaths.from_interval_data converts an IntervalData of "
                "time grids."
            )
        return pixel_breaths
    waveform = as_signal(signal)
    sign_breaths = None
    if breaths is not None:
        global_breaths = _eit_breaths(breaths, action=action)
    elif breath_detector is not None:
        global_breaths = _breaths_used_by(breath_detector, signal, session)
        sign_breaths = pixel_sign_breaths(waveform, breath_detector)
    else:
        raise ValueError(
            f"{action} needs the breaths: connect 'pixel_breaths', 'breaths', "
            "or 'breath_detector' from 'eit.detect_breaths'."
        )
    return find_pixel_breaths(
        pixels, waveform, global_breaths, sign_breaths=sign_breaths
    )


def _pixel_values(
    calculate: Callable[[Signal, list[BreathEvent] | PixelBreaths], IntervalData],
    eit_data: Any,
    signal: Any,
    *,
    timing: str,
    timing_name: str,
    pixel_breaths: Any,
    breaths: Any,
    breath_detector: Any,
    session: M3Session,
    action: str,
) -> IntervalData:
    """Run a pixel TIV or pixel EELI calculation with the chosen breath timing.

    Args:
        calculate (Callable[[Signal, list[BreathEvent] | PixelBreaths],
            IntervalData]): ``pixel_tidal_impedance_variation`` or
            ``pixel_end_expiratory_lung_impedance`` with its settings, taking
            the images and the breaths.
        eit_data (Any): The EIT images (``Signal`` or eitprocessing
            ``EITData``).
        signal (Any): The impedance waveform of the global breaths.
        timing (str): ``"continuous"`` or ``"pixel"``.
        timing_name (str): Name of the timing parameter, for the messages.
        pixel_breaths (Any): The ``PixelBreaths`` input, or None.
        breaths (Any): The ``breaths`` input, or None.
        breath_detector (Any): Breath detection settings, or None.
        session (M3Session): Session with the start times and the stored EIT
            breaths.
        action (str): Step name, for the messages.

    Returns:
        IntervalData: The result of ``calculate``. With given breaths and
            the "continuous" timing, it holds the given breath objects, on
            the clock of their modality. The "continuous" timing leaves
            ``pixel_breaths`` unused.

    Raises:
        ValueError: If ``timing`` is unknown, or the breaths are missing.
    """

    if timing not in _PIXEL_TIMINGS:
        raise ValueError(
            f"{action} '{timing_name}' must be one of "
            f"{', '.join(map(repr, _PIXEL_TIMINGS))}; got {timing!r}."
        )
    pixels = as_signal(eit_data)
    if timing == "pixel":
        return calculate(
            pixels,
            _pixel_breath_timing(
                pixels,
                signal,
                pixel_breaths=pixel_breaths,
                breaths=breaths,
                breath_detector=breath_detector,
                session=session,
                action=action,
            ),
        )
    return _per_breath_values(
        lambda _waveform, found: calculate(pixels, found),
        signal,
        breaths=breaths,
        breath_detector=breath_detector,
        session=session,
        action=action,
    )


def _breath_origin(breaths: Any, pixel_breaths: Any, timing: str) -> dict[str, str]:
    """Return where the breaths of a pixel TIV or EELI step come from.

    Args:
        breaths (Any): The ``breaths`` input, or None.
        pixel_breaths (Any): The ``pixel_breaths`` input, or None.
        timing (str): ``"continuous"`` or ``"pixel"``.

    Returns:
        dict[str, str]: ``{"breaths": ...}`` for the provenance:
            ``"pixel_breaths"``, ``"input"`` or ``"breath_detector"``.
    """

    if pixel_breaths is not None and timing == "pixel":
        return {"breaths": "pixel_breaths"}
    return {"breaths": "breath_detector" if breaths is None else "input"}


def _add_map_metadata(
    result: IntervalData, provenance: dict[str, Any], **fields: Any
) -> None:
    """Add the provenance and the breaths with measured pixels to a pixel result.

    Args:
        result (IntervalData): The pixel TIV or pixel EELI. Its ``metadata``
            gets the new keys.
        provenance (dict[str, Any]): The provenance of the step.
        **fields (Any): More metadata keys, for example the timing.
    """

    pixel_maps = [] if result.values is None else list(result.values)
    valid_breath_indices = [
        index
        for index, pixel_map in enumerate(pixel_maps)
        if not np.all(np.isnan(np.asarray(pixel_map, dtype=float)))
    ]
    result.metadata.update(provenance)
    result.metadata.update(
        {
            "axes": ["row", "column"],
            **fields,
            "valid_breath_indices": valid_breath_indices,
            "valid_breath_fraction": (
                len(valid_breath_indices) / len(pixel_maps) if pixel_maps else 0.0
            ),
        }
    )


_PIXEL_INPUTS: dict[str, str | None] = {
    "eit_data": "filtered_eit",
    "signal": "global_impedance",
    "breath_detector": "breath_detector",
    "breaths": "breaths",
    "pixel_breaths": "pixel_breaths",
    "session": "session",
}
_PIXEL_OPTIONAL = ("breath_detector", "breaths", "pixel_breaths")
_PIXEL_ONE_OF = (("breaths", "breath_detector", "pixel_breaths"),)
_PIXEL_INPUT_ARTIFACTS = (
    _PIXEL_SIGNAL,
    _TIMING_WAVEFORM,
    _BREATH_DETECTOR,
    _GLOBAL_BREATHS,
    _PIXEL_BREATHS,
    _SESSION_ARTIFACT,
)


@register_step(
    "eit.pixel_tiv",
    reads=_PIXEL_INPUTS,
    optional_reads=_PIXEL_OPTIONAL,
    one_of_reads=_PIXEL_ONE_OF,
    writes=("pixel_tiv_result",),
    summary="Compute per-pixel tidal impedance variation (TIV).",
    description="Compute the tidal impedance variation of each pixel in each breath, with m3resp.eit.processing.pixel_tidal_impedance_variation. The timing comes from the global breaths ('continuous') or from the breaths of each pixel ('pixel').",
    category="parameters",
    modality="eit",
    session_writes=("session.interval_data",),
    input_artifacts=_PIXEL_INPUT_ARTIFACTS,
    parameters=(
        _timing_parameter("tiv_timing", "TIV"),
        StepParameter(
            name="tiv_method",
            value_type="choice",
            default="inspiratory",
            choices=TIV_METHODS,
            description="Part of the breath: 'inspiratory' (end of inspiration minus start), 'expiratory' (end of inspiration minus end) or 'mean' (end of inspiration minus the mean of start and end).",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_tivs",
            description="Name of the result.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_tiv_result",
            artifact_type="interval_data",
            description="One pixel map of TIV per breath, each stored with its breath (IntervalData of PixelMap).",
            axes=("breath", "row", "column"),
        ),
    ),
    record=True,
)
def pixel_tiv(
    *,
    eit_data: Any,
    signal: Any,
    session: M3Session,
    breath_detector: Any = None,
    breaths: Any = None,
    pixel_breaths: PixelBreaths | None = None,
    tiv_timing: PixelTiming = "continuous",
    tiv_method: str = "inspiratory",
    result_label: str = "pixel_tivs",
) -> dict[str, Any]:
    """Compute one pixel map of tidal impedance variation (TIV) per breath.

    The per-breath maps are added to ``session.interval_data`` and the step
    is recorded in the session's history. NaNs remain in the maps, including
    maps whose pixels are all missing.

    Args:
        eit_data (Any): EIT images to measure, as a ``Signal`` (or an
            eitprocessing ``EITData``), with time, row and column axes.
        signal (Any): Global or regional waveform of the global breaths.
        session (M3Session): Session that stores the per-breath maps and
            history.
        breath_detector (Any): Settings of the breath detection on
            ``signal`` (``BreathDetectionSettings``), or an eitprocessing
            ``BreathDetection``. Used when no breaths are given.
        breaths (Any): Global breaths (a list of ``BreathEvent``s or an
            ``IntervalData`` of breaths). ``'continuous'`` timing: breaths of
            any modality, moved to the EIT clock with the start times of the
            session. ``'pixel'`` timing: EIT breaths, for the pixel breath
            search.
        pixel_breaths (PixelBreaths | None): The breaths of each pixel from
            ``eit.pixel_breaths``, used by the ``'pixel'`` timing only. In a
            workflow, this is the output ``pixel_breaths`` of an earlier
            ``eit.pixel_breaths`` step. None finds them in the mode
            ``"negative amplitude"``.
        tiv_timing (PixelTiming): ``'continuous'`` uses the times of the
            global breaths; ``'pixel'`` uses the times of the breaths of each
            pixel.
        tiv_method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"``.
        result_label (str): Name of the result.

    Returns:
        dict[str, Any]: ``pixel_tiv_result``, an ``IntervalData`` with a
            row-column ``PixelMap`` of TIV per global breath, in the unit of
            ``eit_data``. Metadata lists the breaths with any measured
            pixels.

    Raises:
        TypeError: If ``pixel_breaths`` is not a ``PixelBreaths``.
        ValueError: If ``tiv_timing`` is unknown, no breaths are given and
            ``breath_detector`` is None, or the ``'pixel'`` timing gets
            breaths of another modality than EIT.
    """

    provenance = _upstream_metadata(
        source_package="m3resp",
        source_function=PIXEL_TIV_FUNCTION,
        implementation=PIXEL_TIV_FUNCTION,
        operation="eit.pixel_tiv",
        parameters={
            "tiv_timing": tiv_timing,
            "tiv_method": tiv_method,
            "result_label": result_label,
            **_breath_origin(breaths, pixel_breaths, tiv_timing),
        },
    )
    result = _pixel_values(
        lambda pixels, timing: pixel_tidal_impedance_variation(
            pixels, timing, method=tiv_method, name=result_label
        ),
        eit_data,
        signal,
        timing=tiv_timing,
        timing_name="tiv_timing",
        pixel_breaths=pixel_breaths,
        breaths=breaths,
        breath_detector=breath_detector,
        session=session,
        action="eit.pixel_tiv",
    )
    _add_map_metadata(
        result, provenance, tiv_timing=tiv_timing, result_label=result_label
    )
    session.interval_data.add(result)
    return {"pixel_tiv_result": result, "_provenance": provenance}


@register_step(
    "eit.pixel_eeli",
    reads=_PIXEL_INPUTS,
    optional_reads=_PIXEL_OPTIONAL,
    one_of_reads=_PIXEL_ONE_OF,
    writes=("pixel_eeli_result",),
    summary="Compute per-pixel end-expiratory lung impedance (EELI).",
    description="Compute the end-expiratory lung impedance of each pixel in each breath, with m3resp.eit.processing.pixel_end_expiratory_lung_impedance: the impedance of the pixel at the end of the breath. The timing comes from the global breaths ('continuous') or from the breaths of each pixel ('pixel').",
    category="parameters",
    modality="eit",
    session_writes=("session.interval_data",),
    input_artifacts=_PIXEL_INPUT_ARTIFACTS,
    parameters=(
        _timing_parameter("eeli_timing", "EELI"),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_eelis",
            description="Name of the result.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_eeli_result",
            artifact_type="interval_data",
            description="One pixel map of EELI per breath, each stored with its breath (IntervalData of PixelMap).",
            axes=("breath", "row", "column"),
        ),
    ),
    record=True,
)
def pixel_eeli(
    *,
    eit_data: Any,
    signal: Any,
    session: M3Session,
    breath_detector: Any = None,
    breaths: Any = None,
    pixel_breaths: PixelBreaths | None = None,
    eeli_timing: PixelTiming = "continuous",
    result_label: str = "pixel_eelis",
) -> dict[str, Any]:
    """Compute one pixel map of end-expiratory lung impedance (EELI) per breath.

    The EELI of a pixel is its impedance at the end of the breath. The
    per-breath maps are added to ``session.interval_data`` and the step is
    recorded in the session's history.

    Args:
        eit_data (Any): EIT images to measure, as a ``Signal`` (or an
            eitprocessing ``EITData``), with time, row and column axes.
        signal (Any): Global or regional waveform of the global breaths.
        session (M3Session): Session that stores the per-breath maps and
            history.
        breath_detector (Any): Settings of the breath detection on
            ``signal`` (``BreathDetectionSettings``), or an eitprocessing
            ``BreathDetection``. Used when no breaths are given.
        breaths (Any): Global breaths, as for ``eit.pixel_tiv``.
        pixel_breaths (PixelBreaths | None): The breaths of each pixel from
            ``eit.pixel_breaths``, used by the ``'pixel'`` timing only. In a
            workflow, this is the output ``pixel_breaths`` of an earlier
            ``eit.pixel_breaths`` step. None finds them in the mode
            ``"negative amplitude"``.
        eeli_timing (PixelTiming): ``'continuous'`` uses the end of the
            global breaths; ``'pixel'`` uses the end of the breath of each
            pixel.
        result_label (str): Name of the result.

    Returns:
        dict[str, Any]: ``pixel_eeli_result``, an ``IntervalData`` with a
            row-column ``PixelMap`` of EELI per global breath, in the unit of
            ``eit_data``. Metadata lists the breaths with any measured
            pixels.

    Raises:
        TypeError: If ``pixel_breaths`` is not a ``PixelBreaths``.
        ValueError: If ``eeli_timing`` is unknown, no breaths are given and
            ``breath_detector`` is None, or the ``'pixel'`` timing gets
            breaths of another modality than EIT.
    """

    provenance = _upstream_metadata(
        source_package="m3resp",
        source_function=PIXEL_EELI_FUNCTION,
        implementation=PIXEL_EELI_FUNCTION,
        operation="eit.pixel_eeli",
        parameters={
            "eeli_timing": eeli_timing,
            "result_label": result_label,
            **_breath_origin(breaths, pixel_breaths, eeli_timing),
        },
    )
    result = _pixel_values(
        lambda pixels, timing: pixel_end_expiratory_lung_impedance(
            pixels, timing, name=result_label
        ),
        eit_data,
        signal,
        timing=eeli_timing,
        timing_name="eeli_timing",
        pixel_breaths=pixel_breaths,
        breaths=breaths,
        breath_detector=breath_detector,
        session=session,
        action="eit.pixel_eeli",
    )
    _add_map_metadata(
        result, provenance, eeli_timing=eeli_timing, result_label=result_label
    )
    session.interval_data.add(result)
    return {"pixel_eeli_result": result, "_provenance": provenance}


@register_step(
    "eit.pixel_breaths",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "breaths": "breaths",
        "session": "session",
    },
    optional_reads=("breaths",),
    writes=("pixel_breaths", "pixel_breath_timing_result"),
    summary="Find the start, end of inspiration and end of each pixel's breath.",
    description="Find the start, end of inspiration and end of the breath of each pixel, from the global breaths, with m3resp.eit.processing.find_pixel_breaths.",
    category="detection",
    modality="eit",
    session_writes=("session.interval_data",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to detect pixel breaths on.",
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying overall breath timing.",
        ),
        StepArtifact(
            name="breaths",
            artifact_type=BREATHS_ARTIFACT_TYPE,
            required=False,
            description=(
                "Global EIT breaths, for example the breaths of "
                "'eit.detect_breaths'. A list of BreathEvents or an "
                "IntervalData of breaths. Without them, the step finds the "
                "breaths on 'timing_data' with 'minimum_duration_seconds'."
            ),
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="phase_correction_mode",
            value_type="choice",
            required=False,
            default="negative amplitude",
            choices=PHASE_CORRECTION_MODES,
            description=(
                "Per-pixel phase correction method. The empty option (`null` "
                "in a YAML spec, `None` from Python) is accepted too and means "
                "the same as 'none'."
            ),
        ),
        StepParameter(
            name="minimum_duration_seconds",
            value_type="number",
            default=2 / 3,
            unit="s",
            minimum=0,
            description="Shortest expected global breath, for the breaths found on 'timing_data'.",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_breaths",
            description="Name of the result.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_breaths",
            artifact_type=PIXEL_BREATHS_ARTIFACT_TYPE,
            unit="s",
            description="The global breaths and the start, end of inspiration and end time of the breath of each pixel (PixelBreaths), for 'eit.pixel_tiv' and 'eit.pixel_eeli'.",
            axes=("breath", "row", "column"),
        ),
        StepArtifact(
            name="pixel_breath_timing_result",
            artifact_type="interval_data",
            unit="s",
            description="Per breath, the [start, end of inspiration, end] time of each pixel's breath, stored with the breath it belongs to (IntervalData).",
            axes=("breath", "row", "column", "landmark"),
        ),
    ),
    record=True,
)
def pixel_breaths(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    breaths: Any = None,
    phase_correction_mode: PhaseCorrectionMode = "negative amplitude",
    minimum_duration_seconds: float = 2 / 3,
    result_label: str = "pixel_breaths",
) -> dict[str, Any]:
    """Find the start, end of inspiration and end time of each pixel's breath.

    Timing grids are added to ``session.interval_data`` and the step is
    recorded in the session's history. Each grid belongs to the enclosing
    global breath.

    Args:
        eit_data (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with time, row and column axes.
        timing_data (Any): Global or regional waveform of the global breaths.
        session (M3Session): Session that stores the per-breath timing grids
            and history.
        breaths (Any): Global EIT breaths (a list of ``BreathEvent``s or an
            ``IntervalData`` of breaths) on the time axis of
            ``timing_data``. They also give the sign of each pixel. None
            finds them on ``timing_data``; the sign of each pixel then comes
            from the breaths found with the default settings, as in
            eitprocessing.
        phase_correction_mode (PhaseCorrectionMode): ``'negative
            amplitude'``, ``'phase shift'``, ``'none'`` or ``None``.
            ``None`` selects no phase correction.
        minimum_duration_seconds (float): Shortest expected global breath,
            in seconds, for the breaths found on ``timing_data``.
        result_label (str): Name of the result.

    Returns:
        dict[str, Any]: ``pixel_breaths``, the ``PixelBreaths``, and
            ``pixel_breath_timing_result``, the same times as
            ``IntervalData``: per global breath a grid of shape
            ``(row, column, 3)`` with start, end of inspiration and end
            times in seconds on the input's clock. Missing timings are NaN.

    Raises:
        TypeError: If a given breath is not a ``BreathEvent``.
        ValueError: If the phase correction mode is unknown, or a given
            breath has another modality than EIT.
    """

    waveform = as_signal(timing_data)
    sign_breaths = None
    if breaths is None:
        settings = BreathDetectionSettings(minimum_duration=minimum_duration_seconds)
        global_breaths = _use_stored_eit_breaths(
            find_breaths(waveform, settings), session
        )
        sign_breaths = pixel_sign_breaths(waveform, settings)
    else:
        global_breaths = _eit_breaths(breaths, action="eit.pixel_breaths")
    result = find_pixel_breaths(
        as_signal(eit_data),
        waveform,
        global_breaths,
        phase_correction_mode=phase_correction_mode,
        name=result_label,
        sign_breaths=sign_breaths,
    )

    provenance = _upstream_metadata(
        source_package="m3resp",
        source_function=PIXEL_BREATH_FUNCTION,
        implementation=PIXEL_BREATH_FUNCTION,
        operation="eit.pixel_breaths",
        parameters={
            "phase_correction_mode": phase_correction_mode,
            "minimum_duration_seconds": minimum_duration_seconds,
            "result_label": result_label,
            "breaths": "timing_data" if breaths is None else "input",
        },
    )
    result.metadata.update(provenance)
    valid = result.has_breath
    pixel_breath_timing_result = result.to_interval_data()
    pixel_breath_timing_result.metadata.update(
        {
            "axes": ["row", "column", "landmark"],
            "landmarks": list(PIXEL_BREATH_POINTS),
            "valid_pixel_breath_count": int(valid.sum()),
            "valid_pixel_breath_fraction": float(valid.mean()) if valid.size else 0.0,
        }
    )
    session.interval_data.add(pixel_breath_timing_result)

    return {
        "pixel_breaths": result,
        "pixel_breath_timing_result": pixel_breath_timing_result,
        "_provenance": provenance,
    }
