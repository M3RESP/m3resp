"""Deprecated alias for :mod:`m3resp.workflows.registry`.

See :mod:`m3resp.pipeline` for the deprecation notice and removal target.
"""

from __future__ import annotations

import warnings

from m3resp.workflows.registry import (
    STEP_REGISTRY,
    StepDefinition,
    available_steps,
    get_step,
    register_step,
)

warnings.warn(
    "m3resp.pipeline.registry is deprecated and will be removed no earlier "
    "than m3resp 0.3.0; import from m3resp.workflows.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "STEP_REGISTRY",
    "StepDefinition",
    "available_steps",
    "get_step",
    "register_step",
]
