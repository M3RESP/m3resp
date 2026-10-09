"""Registered pixel-level EIT workflow steps.

Both steps give one result per breath, so both store an `IntervalData` on
``session.interval_data``: per-pixel TIV as one `PixelMap` per breath, and
per-pixel breath timing as one (row, column, landmark) grid per breath.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    breath_intervals_to_breath_events,
    sparse_data_to_interval_data,
)
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, IntervalData
from m3resp.workflows.registry import (
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
    _use_stored_eit_breaths,
)


@register_step(
    "eit.pixel_tiv",
    reads={
        "eit_data": "filtered_eit",
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
    session_writes=("session.interval_data",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description=(
                "EIT pixel signal to compute per-pixel TIV on. Defaults to the "
                "filtered signal, but any pixel signal can be bound here."
            ),
            compatibility_only=True,
        ),
        StepArtifact(
            name="signal",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying breath timing.",
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
            description=(
                "Whether breath timing is taken per-pixel, or from the "
                "continuous waveform bound to 'signal'. That waveform is the "
                "global impedance (all pixels) by default, but a regional "
                "impedance waveform (from an ROI, e.g. a lung region) works too."
            ),
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
            artifact_type="interval_data",
            description="One pixel map of TIV per breath, each stored with its breath (IntervalData of PixelMap).",
            axes=("breath", "row", "column"),
        ),
    ),
)
def pixel_tiv(
    *,
    eit_data: Any,
    signal: Any,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
    tiv_timing: Literal["pixel", "continuous"] = "continuous",
    result_label: str = "pixel_tivs",
) -> dict[str, Any]:
    """Compute one pixel map of tidal impedance variation (TIV) per breath.

    The per-breath maps are added to ``session.interval_data`` and the step
    is recorded in the session's history. NaNs remain in the maps, including
    maps whose pixels are all missing.

    Args:
        eit_data: eitprocessing EIT pixel data to measure, with time, row and
            column axes.
        signal: Global or regional waveform used to find the enclosing breaths.
        eit_sequence: Sequence that stores the upstream result.
        breath_detector: Detector used to identify breaths on ``signal``.
        session: Session that stores the per-breath maps and history.
        tiv_timing: ``'continuous'`` uses waveform breath timing; ``'pixel'``
            uses individual pixel breath timing.
        result_label: Name used for the upstream and per-breath results.

    Returns:
        dict[str, Any]: ``pixel_tiv`` as the upstream result and
            ``pixel_tiv_result`` as ``IntervalData`` containing a row-column
            ``PixelMap`` per breath. Maps retain the upstream impedance unit
            and breath order. Metadata lists breaths with any measured pixels.

    Raises:
        ValueError: If value and breath counts differ or a pixel map has other
            than two dimensions.
    """

    result = session.eit_adapter.compute_pixel_tiv(
        eit_data,
        signal,
        sequence=eit_sequence,
        breath_detector=breath_detector,
        tiv_timing=tiv_timing,
        result_label=result_label,
    )

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.parameters.tidal_impedance_variation."
            "TIV.compute_pixel_parameter"
        ),
        operation="eit.pixel_tiv",
        parameters={"tiv_timing": tiv_timing, "result_label": result_label},
    )
    # With tiv_timing="pixel" each pixel has its own breath timing; those
    # times are the result of 'eit.pixel_breaths', so only the breath each
    # map belongs to is kept here.
    pixel_tiv_result = sparse_data_to_interval_data(
        result,
        _breaths_used_by(breath_detector, signal, session),
        modality="eit",
        method="eitprocessing.TIV",
        metadata=metadata,
        as_pixel_maps=True,
    )
    pixel_maps = (
        [] if pixel_tiv_result.values is None else list(pixel_tiv_result.values)
    )
    valid_breath_indices = [
        index
        for index, pixel_map in enumerate(pixel_maps)
        if not np.all(np.isnan(pixel_map.values))
    ]
    pixel_tiv_result.metadata.update(
        {
            "axes": ["row", "column"],
            "tiv_timing": tiv_timing,
            "result_label": result_label,
            "valid_breath_indices": valid_breath_indices,
            "valid_breath_fraction": (
                len(valid_breath_indices) / len(pixel_maps) if pixel_maps else 0.0
            ),
        }
    )
    session.interval_data.add(pixel_tiv_result)

    _record_step(session, "eit.pixel_tiv", metadata=metadata)
    return {"pixel_tiv": result, "pixel_tiv_result": pixel_tiv_result}


#: The per-pixel phase correction methods, defined once. The declared
#: `choices`, the runtime check and the type hint all derive from this, so a
#: GUI built from `choices` offers exactly what the step accepts. `None` (YAML
#: `null`) is a real option, equivalent to "none".
#: Written flat rather than as `Literal[...] | None` (which PYI061 would
#: prefer) because only the flat form makes `get_args` return the four options
#: as one tuple, which is what `choices` and the runtime check both need.
PhaseCorrectionMode = Literal["negative amplitude", "phase shift", "none", None]  # noqa: PYI061

_ALLOWED_PIXEL_BREATH_PHASE_MODES = get_args(PhaseCorrectionMode)


def _pixel_breaths_to_landmark_array(values: Any) -> np.ndarray:
    """Convert `PixelBreath`'s `(breath, row, column)` object array of
    `Breath | None` into a numeric `(breath, row, column, landmark)` array,
    where landmark is `[start_time, middle_time, end_time]`. Missing pixel
    breaths (including the always-unavailable first/last global breath)
    become NaN."""

    array = np.asarray(values, dtype=object)
    landmarks = np.full((*array.shape, 3), np.nan, dtype=float)
    for breath_index, row, col in np.ndindex(array.shape):
        # Object-dtype element, so a `Breath | None`, not an array.
        breath: Any = array[breath_index, row, col]
        if breath is not None:
            landmarks[breath_index, row, col] = (
                breath.start_time,
                breath.middle_time,
                breath.end_time,
            )
    return landmarks


def _pixel_breath_intervals(
    intervals: Any, breath_intervals: Any, session: M3Session
) -> list[BreathEvent]:
    """Match pixel-breath intervals to detected breaths with turning points.

    ``intervals`` gives start-end pairs in seconds; ``breath_intervals``
    supplies the corresponding middle times. Matching checks exact start and
    end times in the same order and reuses matching stored EIT breaths.

    Raises:
        ValueError: If the breaths found again do not have the same start
            and end times as PixelBreath's intervals.
    """

    breaths = _use_stored_eit_breaths(
        breath_intervals_to_breath_events(breath_intervals), session
    )
    expected = [(float(start), float(end)) for start, end in intervals]
    found = [(breath.start_time, breath.end_time) for breath in breaths]
    if found != expected:
        raise ValueError(
            "eit.pixel_breaths: the breaths found on 'timing_data' do not match "
            f"the {len(expected)} breaths PixelBreath used (found {len(found)})."
        )
    return breaths


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
    session_writes=("session.interval_data",),
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
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying overall breath timing.",
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
            choices=_ALLOWED_PIXEL_BREATH_PHASE_MODES,
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
            artifact_type="interval_data",
            unit="s",
            description="Per breath, the [start, middle, end] time of each pixel's breath, stored with the breath it belongs to (IntervalData).",
            axes=("breath", "row", "column", "landmark"),
        ),
    ),
)
def pixel_breaths(
    *,
    eit_data: Any,
    timing_data: Any,
    eit_sequence: Any,
    session: M3Session,
    phase_correction_mode: PhaseCorrectionMode = "negative amplitude",
    minimum_duration_seconds: float = 2 / 3,
    result_label: str = "pixel_breaths",
) -> dict[str, Any]:
    """Measure start, middle and end times for each pixel's breath.

    Timing grids are added to ``session.interval_data`` and the step is
    recorded in the session's history. Each grid belongs to the enclosing
    breath detected on ``timing_data``.

    Args:
        eit_data: eitprocessing pixel data with time, row and column axes.
        timing_data: Global or regional waveform used to find enclosing breaths.
        eit_sequence: Sequence that stores the upstream result.
        session: Session that stores the per-breath timing grids and history.
        phase_correction_mode: ``'negative amplitude'``, ``'phase shift'``,
            ``'none'`` or ``None``. ``None`` selects no phase correction.
        minimum_duration_seconds: Minimum duration used for breath detection,
            in seconds.
        result_label: Name used for the upstream and per-breath results.

    Returns:
        dict[str, Any]: ``pixel_breaths`` as the upstream result and
            ``pixel_breath_timing_result`` as ``IntervalData``. Each breath has
            a grid of shape ``(row, column, 3)`` with start, middle and end
            times in seconds on the input's clock. Missing timings are NaN;
            validity counts require all three times to be present.

    Raises:
        ValueError: If the phase correction mode is unknown, or the detected
            breath start-end times differ from the pixel result's intervals.
    """

    if phase_correction_mode not in _ALLOWED_PIXEL_BREATH_PHASE_MODES:
        named = ", ".join(
            repr(mode) for mode in _ALLOWED_PIXEL_BREATH_PHASE_MODES if mode is not None
        )
        raise ValueError(
            "eit.pixel_breaths 'phase_correction_mode' must be one of "
            f"{named}, or empty (`null` in a YAML spec, `None` from Python); "
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
    # A pixel breath counts as determined only when all three of its timings
    # are present. Checking the start time alone would be enough for a breath
    # this module built itself, which sets all three together or none, but it
    # would silently pass a breath whose middle or end time came back missing.
    valid = ~np.isnan(landmarks).any(axis=-1)

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
            "axes": ["row", "column", "landmark"],
            "landmarks": ["start_time", "middle_time", "end_time"],
            "valid_pixel_breath_count": int(valid.sum()),
            "valid_pixel_breath_fraction": float(valid.mean()) if valid.size else 0.0,
        }
    )
    breaths = _pixel_breath_intervals(
        result.intervals,
        session.eit_adapter.find_breaths(
            timing_data, minimum_duration_seconds=minimum_duration_seconds
        ),
        session,
    )
    pixel_breath_timing_result = IntervalData(
        name=result_label,
        modality="eit",
        intervals=list(breaths),
        values=list(landmarks),
        unit="s",
        method="eitprocessing.PixelBreath",
        metadata=metadata,
    )
    session.interval_data.add(pixel_breath_timing_result)

    _record_step(session, "eit.pixel_breaths", metadata=metadata)
    return {
        "pixel_breaths": result,
        "pixel_breath_timing_result": pixel_breath_timing_result,
    }
