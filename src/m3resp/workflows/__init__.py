"""Declarative pipeline engine for M3Resp.

Workflows are described as ordered lists of named steps in a YAML or JSON spec
and executed by :func:`run_pipeline`, without writing custom Python per workflow.
"""

from m3resp.workflows.context import PipelineContext
from m3resp.workflows.engine import (
    PipelineResult,
    run_pipeline,
    run_spec,
    validate_spec,
)
from m3resp.workflows.registry import (
    STEP_REGISTRY,
    StepDefinition,
    available_steps,
    get_step,
    register_step,
)
from m3resp.workflows.spec import PipelineSpec, StepSpec, load_spec

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
