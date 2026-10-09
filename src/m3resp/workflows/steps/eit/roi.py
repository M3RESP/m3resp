"""Registered region-of-interest (ROI) EIT workflow steps."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

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
    _record_step,
    _upstream_metadata,
)


def _validate_unit_threshold(value: float, *, step: str, param: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{step} '{param}' must be between 0 and 1, got {value!r}.")


def _to_pixel_mask(
    mask: Any, *, name: str, method: str, metadata: dict[str, Any]
) -> PixelMask:
    """Convert an upstream mask to a row-column ``PixelMask`` with EIT metadata.

    NaN marks excluded pixels. The values are copied unchanged, zeros and
    out-of-range weights included, since the upstream mask has already run
    the same checks. Metadata records the axes and the count and fraction of
    included pixels.
    """

    values = np.asarray(mask.mask, dtype=float)
    included = ~np.isnan(values)
    metadata = dict(metadata)
    metadata.update(
        {
            "axes": ["row", "column"],
            "included_pixel_count": int(included.sum()),
            "included_pixel_fraction": float(included.mean()) if values.size else 0.0,
        }
    )
    return PixelMask(
        name=name,
        values=values,
        modality="eit",
        method=method,
        metadata=metadata,
        keep_zeros=True,
        suppress_value_range_error=True,
        suppress_all_nan_warning=True,
    )


@register_step(
    "eit.roi_tiv_lungspace",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=("tiv_lungspace_mask", "tiv_lungspace_captures", "tiv_lungspace_result"),
    summary="Threshold mean pixel TIV into a functional lung-space mask.",
    description="Threshold mean per-pixel TIV into a boolean/NaN-excluded functional lung-space mask via eitprocessing's TIVLungspace.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Fraction of the maximum mean pixel TIV a pixel must reach to be included. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="tiv_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="tiv_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="tiv_lungspace_result",
            artifact_type="pixel_mask",
            description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out.",
            axes=("row", "column"),
        ),
    ),
)
def roi_tiv_lungspace(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Select lung-space pixels by their mean tidal impedance variation (TIV).

    The mask is added to ``session.pixel_masks`` and the processing settings
    are recorded in the session's history.

    Args:
        eit_data: eitprocessing pixel data with time, row and column axes.
        timing_data: Waveform used to detect breaths for mean pixel TIV.
        session: Session that stores the mask and processing history.
        threshold: Fraction of maximum mean pixel TIV required for inclusion,
            strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``tiv_lungspace_mask`` as the upstream mask,
            ``tiv_lungspace_captures`` with intermediate results, and
            ``tiv_lungspace_result`` as an m3resp ``PixelMask``. Grids have
            row-column order, with 1 for included pixels and NaN elsewhere.

    Raises:
        ValueError: If the threshold is outside the open interval (0, 1).
    """

    _validate_unit_threshold(threshold, step="eit.roi_tiv_lungspace", param="threshold")

    result = session.eit_adapter.compute_tiv_lungspace(
        eit_data, timing_data=timing_data, threshold=threshold
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.tiv.TIVLungspace.apply",
        operation="eit.roi_tiv_lungspace",
        parameters={"threshold": threshold},
    )
    tiv_lungspace_result = _to_pixel_mask(
        mask,
        name="tiv_lungspace_mask",
        method="eitprocessing.TIVLungspace",
        metadata=dict(metadata),
    )
    session.pixel_masks.add(tiv_lungspace_result)

    _record_step(session, "eit.roi_tiv_lungspace", metadata=metadata)
    return {
        "tiv_lungspace_mask": mask,
        "tiv_lungspace_captures": result["captures"],
        "tiv_lungspace_result": tiv_lungspace_result,
    }


@register_step(
    "eit.roi_amplitude_lungspace",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=(
        "amplitude_lungspace_mask",
        "amplitude_lungspace_captures",
        "amplitude_lungspace_result",
    ),
    summary=(
        "Threshold mean pixel amplitude into a lung-space mask (not "
        "recommended alone; supports eit.roi_watershed)."
    ),
    description=(
        "Threshold mean pixel amplitude into a lung-space mask via "
        "eitprocessing's AmplitudeLungspace. Not recommended as a "
        "general-purpose functional lung-space definition on its own "
        "(may include reconstruction artifacts); intended to support "
        "eit.roi_watershed."
    ),
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Fraction of the maximum mean pixel amplitude a pixel must reach to be included. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="amplitude_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="amplitude_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="amplitude_lungspace_result",
            artifact_type="pixel_mask",
            description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out.",
            axes=("row", "column"),
        ),
    ),
)
def roi_amplitude_lungspace(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Threshold mean pixel amplitude into a lung-space mask.

    Warning: upstream does not recommend amplitude alone as a general-purpose
    functional lung-space definition, as it potentially includes
    reconstruction artifacts. It is provided primarily for use by
    `eit.roi_watershed`.

    The mask is added to ``session.pixel_masks`` and the processing settings
    are recorded in the session's history.

    Args:
        eit_data: eitprocessing pixel data with time, row and column axes.
        timing_data: Waveform used to detect breaths for mean pixel amplitude.
        session: Session that stores the mask and processing history.
        threshold: Fraction of maximum mean pixel amplitude required for
            inclusion, strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``amplitude_lungspace_mask`` as the upstream mask,
            ``amplitude_lungspace_captures`` with intermediate results, and
            ``amplitude_lungspace_result`` as an m3resp ``PixelMask``. Grids
            have row-column order, with 1 for included pixels and NaN elsewhere.

    Raises:
        ValueError: If the threshold is outside the open interval (0, 1).
    """

    _validate_unit_threshold(
        threshold, step="eit.roi_amplitude_lungspace", param="threshold"
    )

    result = session.eit_adapter.compute_amplitude_lungspace(
        eit_data, timing_data=timing_data, threshold=threshold
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.amplitude.AmplitudeLungspace.apply",
        operation="eit.roi_amplitude_lungspace",
        parameters={"threshold": threshold},
    )
    amplitude_lungspace_result = _to_pixel_mask(
        mask,
        name="amplitude_lungspace_mask",
        method="eitprocessing.AmplitudeLungspace",
        metadata=dict(metadata),
    )
    session.pixel_masks.add(amplitude_lungspace_result)

    _record_step(session, "eit.roi_amplitude_lungspace", metadata=metadata)
    return {
        "amplitude_lungspace_mask": mask,
        "amplitude_lungspace_captures": result["captures"],
        "amplitude_lungspace_result": amplitude_lungspace_result,
    }


@register_step(
    "eit.roi_watershed",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=(
        "watershed_lungspace_mask",
        "watershed_captures",
        "watershed_lungspace_result",
    ),
    summary="Derive a lung-space mask with the watershed method (pendelluft-aware).",
    description="Derive a lung-space mask via eitprocessing's watershed method, which stays pendelluft-aware unlike a simple threshold.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_impedance_waveform",
            default_context_key="global_impedance",
            description="Impedance waveform (global, or regional from an ROI) supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold_fraction",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Watershed threshold fraction. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="watershed_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="watershed_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="watershed_lungspace_result",
            artifact_type="pixel_mask",
            description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out.",
            axes=("row", "column"),
        ),
    ),
)
def roi_watershed(
    *,
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    threshold_fraction: float = 0.15,
) -> dict[str, Any]:
    """Select lung-space pixels with eitprocessing's watershed method.

    The method uses mean TIV and amplitude to select image regions. The
    resulting mask is added to ``session.pixel_masks`` and the processing
    settings are recorded in the session's history.

    Args:
        eit_data: eitprocessing pixel data with time, row and column axes.
        timing_data: Waveform used to detect breaths for the pixel calculations.
        session: Session that stores the mask and processing history.
        threshold_fraction: Fraction of maximum mean pixel TIV used for the
            initial mask, strictly between 0 and 1.

    Returns:
        dict[str, Any]: ``watershed_lungspace_mask`` as the upstream mask,
            ``watershed_captures`` with intermediate results, and
            ``watershed_lungspace_result`` as an m3resp ``PixelMask``. Grids
            have row-column order, with 1 for included pixels and NaN elsewhere.

    Raises:
        ValueError: If the threshold is outside the open interval (0, 1).
    """

    _validate_unit_threshold(
        threshold_fraction, step="eit.roi_watershed", param="threshold_fraction"
    )

    result = session.eit_adapter.compute_watershed_lungspace(
        eit_data, timing_data=timing_data, threshold_fraction=threshold_fraction
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.watershed.WatershedLungspace.apply",
        operation="eit.roi_watershed",
        parameters={"threshold_fraction": threshold_fraction},
    )
    watershed_lungspace_result = _to_pixel_mask(
        mask,
        name="watershed_lungspace_mask",
        method="eitprocessing.WatershedLungspace",
        metadata=dict(metadata),
    )
    session.pixel_masks.add(watershed_lungspace_result)

    _record_step(session, "eit.roi_watershed", metadata=metadata)
    return {
        "watershed_lungspace_mask": mask,
        "watershed_captures": result["captures"],
        "watershed_lungspace_result": watershed_lungspace_result,
    }


@register_step(
    "eit.roi_filter_by_size",
    reads={"mask": None, "session": "session"},
    writes=("size_filtered_roi_mask", "size_filtered_roi_result"),
    summary="Keep only connected mask regions at or above a minimum size.",
    description="Drop connected regions of a lung-space mask smaller than a minimum pixel count, via eitprocessing's FilterROIBySize.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.pixel_masks",),
    input_artifacts=(
        StepArtifact(
            name="mask",
            artifact_type=ANY_ARTIFACT_TYPE,
            description=(
                "Lung-space mask to filter. Either form works: the upstream "
                "mask a mask step writes (e.g. 'watershed_lungspace_mask') or "
                "its native counterpart (e.g. 'watershed_lungspace_result')."
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
            description="Minimum connected-region size kept.",
        ),
        StepParameter(
            name="connectivity",
            value_type="integer",
            default=1,
            choices=(1, 2),
            description="Pixel connectivity: 1 (4-connected) or 2 (8-connected).",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="size_filtered_roi_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask with small regions dropped.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="size_filtered_roi_result",
            artifact_type="pixel_mask",
            description="m3resp PixelMask (row, column): 1 for pixels in the region, NaN for pixels left out.",
            axes=("row", "column"),
        ),
    ),
)
def roi_filter_by_size(
    mask: Any,
    *,
    session: M3Session,
    min_region_size: int = 10,
    connectivity: Literal[1, 2] = 1,
) -> dict[str, Any]:
    """Remove mask regions smaller than the specified number of pixels.

    The filtered mask is added to ``session.pixel_masks`` and the processing
    settings are recorded in the session's history.

    Args:
        mask: An eitprocessing or m3resp ``PixelMask``, or a 2D numeric grid
            with NaN for excluded pixels.
        session: Session that stores the filtered mask and processing history.
        min_region_size: Minimum number of connected pixels to retain; positive.
        connectivity: ``1`` joins edge neighbours; ``2`` also joins diagonals.

    Returns:
        dict[str, Any]: ``size_filtered_roi_mask`` as the upstream mask and
            ``size_filtered_roi_result`` as an m3resp ``PixelMask``, with
            row-column order and NaN for excluded pixels.

    Raises:
        ValueError: If ``min_region_size`` is zero or negative.
        UnsupportedWorkflowError: If the input cannot be converted to a
            2D numeric mask.
    """

    if min_region_size <= 0:
        raise ValueError(
            "eit.roi_filter_by_size 'min_region_size' must be positive, "
            f"got {min_region_size!r}."
        )

    result = session.eit_adapter.filter_roi_by_size(
        mask, min_region_size=min_region_size, connectivity=connectivity
    )
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.filter_by_size.FilterROIBySize.apply",
        operation="eit.roi_filter_by_size",
        parameters={
            "min_region_size": min_region_size,
            "connectivity": connectivity,
        },
    )
    size_filtered_roi_result = _to_pixel_mask(
        result,
        name="size_filtered_roi_mask",
        method="eitprocessing.FilterROIBySize",
        metadata=dict(metadata),
    )
    session.pixel_masks.add(size_filtered_roi_result)

    _record_step(session, "eit.roi_filter_by_size", metadata=metadata)
    return {
        "size_filtered_roi_mask": result,
        "size_filtered_roi_result": size_filtered_roi_result,
    }
