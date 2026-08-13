"""Time-offset estimation between simultaneously recorded modalities.

When two devices record the same subject on independent clocks -- for example a
Draeger EIT device (~50 Hz frame rate) and a Biopac amplifier (2 kHz) carrying
airway pressure (Paw) plus diaphragm sEMG -- the recordings share no common
timestamp and must be aligned before analysis. :meth:`M3Session.
synchronize_raw_modalities` *applies* a known offset (it crops the modalities
onto a common window), but it does not *find* it. This module supplies the
missing estimation stage.

There is currently no robust, general-purpose automatic sync method, so the
only supported method is ``"manual"``: the offset is supplied by the caller,
typically found interactively with the marimo multimodal viewer
(``tools/visualization_tools/2_annemijn_multimodal_vis.py``) and its
protocol-specific estimators
(``tools/visualization_tools/utils/offset_estimation.py``), then hardcoded
into the pipeline spec. Those estimators are not part of this module because
they were tuned against one dataset's specific acquisition artifact and are
not safe to run unattended on arbitrary recordings.

Offset convention
-----------------
Throughout this module an *offset* is the timestamp, **on the reference clock**,
at which the target recording's ``t = 0`` occurs. Equivalently, a sample at
target-relative time ``t`` maps to reference time ``offset + t``. For the EIT +
Biopac case the reference clock is the Biopac timeline and the target is the EIT
recording, so ``offset`` is "Biopac seconds at EIT t=0".
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SyncOffsetResult",
    "estimate_sync_offset",
]

#: Methods accepted by :func:`estimate_sync_offset`.
_METHODS = frozenset({"manual"})


@dataclass(frozen=True)
class SyncOffsetResult:
    """Result returned by :func:`estimate_sync_offset`."""

    offset_seconds: float
    method: str
    source: str


def estimate_sync_offset(
    *,
    method: str = "manual",
    manual_offset_seconds: float = 0.0,
) -> SyncOffsetResult:
    """Return a manually supplied sync offset.

    ``"manual"`` is currently the only supported method: there is no robust,
    general-purpose automatic sync estimator in the installable package (see
    the module docstring for where the protocol-specific estimators live
    instead).
    """

    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {sorted(_METHODS)}"
        )

    return SyncOffsetResult(
        offset_seconds=float(manual_offset_seconds),
        method=method,
        source="manual offset",
    )
