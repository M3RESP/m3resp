"""Registered ventilator pipeline steps.

Split out of `m3resp.workflows.steps.emg` once the ventilator became a peer
modality (its own `VentilatorAdapter`, its own `M3Session.load_ventilator`/
`preprocess_ventilator`/`detect_ventilator_breaths`) rather than a passenger of
the EMG path. Every former `emg.*` id these steps used to be registered under
still resolves - see `register_step`'s `aliases=` and `m3resp.workflows.
registry.STEP_ALIASES` - so existing pipeline specs keep running unchanged.

Mirrors `m3resp.workflows.steps.emg`'s package layout: importing this module
registers every step below (each submodule's `@register_step` decorators run
on import), and every public step function is re-exported here so
`from m3resp.workflows.steps.ventilator import <name>` works.
"""

from __future__ import annotations

from .detection import (
    detect_breaths,
    find_occluded_breaths,
    pocc_intervals,
    pocc_quality,
    pocc_time_product,
)
from .features import respiratory_rate
from .loading import channels, load
from .normalization import normalize_breaths
from .quality import detect_non_consecutive_manoeuvres

__all__ = [
    "channels",
    "detect_breaths",
    "detect_non_consecutive_manoeuvres",
    "find_occluded_breaths",
    "load",
    "normalize_breaths",
    "pocc_intervals",
    "pocc_quality",
    "pocc_time_product",
    "respiratory_rate",
]
