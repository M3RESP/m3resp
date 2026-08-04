"""Registered EIT pipeline steps.

Each step wraps a single ``eitprocessing`` operation. Upstream imports are
deferred to call time so the package installs without the optional
``eitprocessing`` dependency.

This package mirrors the former single ``eit.py`` module, split by pipeline
stage for readability. Importing it registers every step below (each
submodule's ``@register_step`` decorators run on import), and every public
step function is re-exported here so ``from m3resp.workflows.steps.eit
import <name>`` keeps working unchanged.
"""

from __future__ import annotations

from .filtering import butterworth_filter, detect_rates, mdn_filter
from .loading import load, slice_signal
from .pixel import _ALLOWED_PIXEL_BREATH_PHASE_MODES, pixel_breaths, pixel_tiv
from .roi import (
    roi_amplitude_lungspace,
    roi_filter_by_size,
    roi_tiv_lungspace,
    roi_watershed,
)
from .signals import (
    continuous_tiv,
    detect_breaths,
    eeli,
    global_impedance,
    normalize_breaths,
)

__all__ = [
    "_ALLOWED_PIXEL_BREATH_PHASE_MODES",
    "butterworth_filter",
    "continuous_tiv",
    "detect_breaths",
    "detect_rates",
    "eeli",
    "global_impedance",
    "load",
    "mdn_filter",
    "normalize_breaths",
    "pixel_breaths",
    "pixel_tiv",
    "roi_amplitude_lungspace",
    "roi_filter_by_size",
    "roi_tiv_lungspace",
    "roi_watershed",
    "slice_signal",
]
