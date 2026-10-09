"""Registered EMG step that removes breaths which failed chosen quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step
from m3resp.workflows.steps._per_breath import extremum_sample_index

from ._shared import _SESSION_ARTIFACT, _record_step, _upstream_metadata

# Per-breath EMG features this step shortens together with the breaths.
# Each one is an array with one value per breath, or a pair of such arrays
# (time-to-peak in seconds and in percent; area under baseline and its
# reference values).
_FEATURE_KEYS = (
    "time_to_peak",
    "pseudo_slope",
    "amplitude",
    "time_product",
    "area_under_baseline",
)


@register_step(
    "emg.remove_invalid_breaths",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        **{key: key for key in _FEATURE_KEYS},
    },
    optional_reads=("start_indices", "end_indices", *_FEATURE_KEYS),
    writes=("valid_breaths",),
    summary="Keep only the EMG breaths that passed chosen quality checks.",
    description="Remove the EMG breaths that failed any of the chosen per-breath quality checks, and return the remaining breaths with their onsets, offsets and features. The original per-breath outputs are left unchanged. Comparable to ReSurfEMG's PeaksSet.sanitize, but the checks that decide are named explicitly.",
    category="quality",
    modality="emg",
    session_reads=("session.quality",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices, if computed.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices, if computed.",
        ),
        *(
            StepArtifact(
                name=key,
                artifact_type="array",
                description=f"Per-breath '{key}' from 'emg.{key}', if computed.",
            )
            for key in _FEATURE_KEYS
        ),
    ),
    parameters=(
        StepParameter(
            name="flag_names",
            value_type="list",
            default=("start_end_validity",),
            description="Names of the per-breath EMG quality flags that decide. A breath is removed if any of them failed for it. For example 'start_end_validity', 'evaluate_bell_curve_error', 'snr_pseudo'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="valid_breaths",
            artifact_type="mapping",
            description="Kept breaths: their original breath numbers, peak/onset/offset indices and features, plus a list of the removed breaths with the flags they failed.",
        ),
    ),
)
def remove_invalid_breaths(
    session: M3Session,
    peak_indices: Any,
    *,
    start_indices: Any = None,
    end_indices: Any = None,
    time_to_peak: Any = None,
    pseudo_slope: Any = None,
    amplitude: Any = None,
    time_product: Any = None,
    area_under_baseline: Any = None,
    flag_names: Any = ("start_end_validity",),
) -> dict[str, Any]:
    """Remove EMG breaths that do not pass the selected quality checks.

    Checks are matched by turning-point sample position. For each check and
    sample, the latest flag in session.quality is used. A breath is removed
    if any selected check failed; breaths a check did not assess are kept.
    Reading older ``peak_sample_index`` metadata emits a UserWarning.
    Records the selection settings in the session's processing history.

    Args:
        session: Session holding per-breath EMG quality flags.
        peak_indices: EMG envelope peak positions, one per breath.
        start_indices: Optional onset positions, in the same breath order.
        end_indices: Optional offset positions, in the same breath order.
        time_to_peak: Optional pair of per-breath arrays, in seconds and percent.
        pseudo_slope: Optional per-breath pseudo-slopes, in envelope units
            per sample.
        amplitude: Optional per-breath amplitudes, in envelope units.
        time_product: Optional per-breath integrals, in envelope units times seconds.
        area_under_baseline: Optional pair of per-breath arrays containing
            areas in envelope units times seconds and reference signal
            nadir values in envelope units.
        flag_names: Names of the quality checks used to decide which breaths
            to keep; must contain at least one name.

    Returns:
        dict[str, Any]: A ``valid_breaths`` mapping containing original zero-based
            ``breath_numbers``, kept peak/start/end indices, kept feature arrays
            and flag names. Missing start/end arrays remain None. Each removed
            breath is listed with its original number, ``extremum_sample_index``
            and failed flag names. Supplied arrays and session events are
            preserved.

    Raises:
        ValueError: If flag_names is empty, a selected EMG check is missing or
            has a flag without a turning-point sample, or a supplied array has
            a different number of rows from peak_indices.
    """

    peaks = np.asarray(peak_indices, dtype=int)
    names = [str(name) for name in flag_names]
    if not names:
        raise ValueError("'flag_names' must name at least one quality flag.")

    failed_flags = _failed_flags_per_breath(session, peaks, names)
    keep = np.array([not failed for failed in failed_flags], dtype=bool)

    features = {
        "time_to_peak": time_to_peak,
        "pseudo_slope": pseudo_slope,
        "amplitude": amplitude,
        "time_product": time_product,
        "area_under_baseline": area_under_baseline,
    }
    kept_features = {
        key: _keep_rows(key, value, keep)
        for key, value in features.items()
        if value is not None
    }

    valid_breaths: dict[str, Any] = {
        # Position of each kept breath in the original breath list, so it
        # can still be matched to outputs that were not shortened.
        "breath_numbers": np.flatnonzero(keep),
        "peak_indices": peaks[keep],
        "start_indices": _keep_rows("start_indices", start_indices, keep),
        "end_indices": _keep_rows("end_indices", end_indices, keep),
        "features": kept_features,
        "flag_names": names,
        "removed": [
            {
                "breath_number": int(position),
                "extremum_sample_index": int(peaks[position]),
                "failed_flags": failed_flags[position],
            }
            for position in np.flatnonzero(~keep)
        ],
    }

    _record_step(
        session,
        "emg.remove_invalid_breaths",
        metadata=_upstream_metadata(
            source_function="m3resp.workflows.steps.emg.selection.remove_invalid_breaths",
            operation="emg.remove_invalid_breaths",
            parameters={"flag_names": names},
            source_package="m3resp",
            implementation="m3resp.workflows.steps.emg.selection",
        ),
    )
    return {"valid_breaths": valid_breaths}


def _failed_flags_per_breath(
    session: M3Session, peaks: np.ndarray, names: list[str]
) -> list[list[str]]:
    """Return the selected failed check names for each breath in input order.

    Match EMG flags by ``extremum_sample_index``, reading the older
    ``peak_sample_index`` with a UserWarning when needed. The latest flag for
    each check and sample is used. Breaths missing a check's flag are kept.
    Raises ValueError if a selected check is missing or has a flag without
    a turning-point sample.
    """

    position_of_peak = {int(peak): position for position, peak in enumerate(peaks)}
    failed: list[list[str]] = [[] for _ in peaks]
    for name in names:
        latest: dict[int, bool] = {}
        for flag in session.quality:
            if flag.name != name or flag.modality != "emg":
                continue
            peak = extremum_sample_index(flag.metadata)
            if peak is None:
                raise ValueError(
                    f"Quality flag '{name}' is not a per-breath flag, so it "
                    "cannot be used to remove breaths."
                )
            latest[int(peak)] = flag.passed
        if not latest:
            raise ValueError(
                f"No EMG quality flag named '{name}' was found. Run the step "
                "that produces it before 'emg.remove_invalid_breaths'."
            )
        for peak, passed in latest.items():
            position = position_of_peak.get(peak)
            if position is not None and not passed:
                failed[position].append(name)
    return failed


def _keep_rows(name: str, value: Any, keep: np.ndarray) -> Any:
    """Keep the rows of a per-breath array (or pair of arrays) where `keep`
    is True. Raises if the array does not have one row per breath."""

    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_keep_rows(name, part, keep) for part in value)
    array = np.asarray(value)
    if len(array) != len(keep):
        raise ValueError(
            f"'{name}' has {len(array)} values but there are {len(keep)} "
            "breaths; they must match one to one."
        )
    return array[keep]
