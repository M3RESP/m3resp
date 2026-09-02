"""Declarative pipeline engine for M3Resp.

Workflows are described as ordered lists of named steps in a YAML or JSON spec
and executed by :func:`run_pipeline`, without writing custom Python per workflow.
"""

from m3resp.workflows.compiler import (
    CompiledPipeline,
    CompiledStep,
    ValidationReport,
    compile_pipeline,
    validate_pipeline,
)
from m3resp.workflows.context import RESOLVED_OUTPUT_DIR_KEY, PipelineContext
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.engine import (
    PipelineResult,
    collect_diagnostics,
    run_pipeline,
    run_spec,
    validate_spec,
)
from m3resp.workflows.lifecycle import (
    CancellationToken,
    CapturedWarning,
    ExecutionContext,
    PipelineExecutionError,
    StepExecutionRecord,
)
from m3resp.workflows.registry import (
    STEP_ALIASES,
    STEP_REGISTRY,
    StepDefinition,
    available_steps,
    get_step,
    register_step,
)
from m3resp.workflows.service import PipelineService, summarize_pipeline_result
from m3resp.workflows.spec import PipelineSpec, StepSpec, load_spec

__all__ = [
    "RESOLVED_OUTPUT_DIR_KEY",
    "STEP_ALIASES",
    "STEP_REGISTRY",
    "CancellationToken",
    "CapturedWarning",
    "CompiledPipeline",
    "CompiledStep",
    "Diagnostic",
    "ExecutionContext",
    "PipelineContext",
    "PipelineExecutionError",
    "PipelineResult",
    "PipelineService",
    "PipelineSpec",
    "StepDefinition",
    "StepExecutionRecord",
    "StepSpec",
    "ValidationReport",
    "available_steps",
    "collect_diagnostics",
    "compile_pipeline",
    "get_step",
    "load_spec",
    "register_step",
    "run_pipeline",
    "run_spec",
    "summarize_pipeline_result",
    "validate_pipeline",
    "validate_spec",
]
