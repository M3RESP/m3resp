"""Registered EIT workflow steps.

Each step wraps a single ``eitprocessing`` operation. Upstream imports are
deferred to call time so the package installs without the optional
``eitprocessing`` dependency.

This package mirrors the former single ``eit.py`` module, split by workflow
stage for readability. Importing it registers every step below (each
submodule's ``@register_step`` decorators run on import), and every public
step function is re-exported here so ``from m3resp.workflows.steps.eit
import <name>`` keeps working unchanged.
"""

from __future__ import annotations

from .filtering import butterworth_filter, mdn_filter
from .loading import load, load_custom
from .pixel import pixel_breaths, pixel_eeli, pixel_tiv
from .rates import detect_rates
from .roi import (
    geometric_mask,
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
from .slicing import slice_recording, slice_signal

__all__ = [
    "butterworth_filter",
    "continuous_tiv",
    "detect_breaths",
    "detect_rates",
    "eeli",
    "geometric_mask",
    "global_impedance",
    "load",
    "load_custom",
    "mdn_filter",
    "normalize_breaths",
    "pixel_breaths",
    "pixel_eeli",
    "pixel_tiv",
    "roi_amplitude_lungspace",
    "roi_filter_by_size",
    "roi_tiv_lungspace",
    "roi_watershed",
    "slice_recording",
    "slice_signal",
]
