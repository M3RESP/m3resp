"""Deprecated compatibility shim for the earlier ``m3resp.pipeline`` import path.

``m3resp.workflows`` is the canonical Stage 2/Stage 3 pipeline module (see
``plan/stage2/3_pipeline_structure_implementation_plan.md``, "Decisions Made
Up Front"). This package re-exports the same public API so old imports keep
working, and will be removed no earlier than ``m3resp`` 0.3.0. New code
should import from ``m3resp.workflows`` directly.
"""

from __future__ import annotations

import warnings

from m3resp.workflows import (
    PipelineContext,
    PipelineResult,
    PipelineSpec,
    STEP_REGISTRY,
    StepDefinition,
    StepSpec,
    available_steps,
    get_step,
    load_spec,
    register_step,
    run_pipeline,
    run_spec,
    validate_spec,
)

warnings.warn(
    "m3resp.pipeline is deprecated and will be removed no earlier than "
    "m3resp 0.3.0; import from m3resp.workflows instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "STEP_REGISTRY",
    "PipelineContext",
    "PipelineResult",
    "PipelineSpec",
    "StepDefinition",
    "StepSpec",
    "available_steps",
    "get_step",
    "load_spec",
    "register_step",
    "run_pipeline",
    "run_spec",
    "validate_spec",
]
