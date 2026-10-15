"""Declarative workflow engine for M3Resp.

Workflows are described as ordered lists of named steps in a YAML or JSON spec
and executed by :func:`run_workflow`, without writing custom Python per workflow.
"""

from m3resp.workflows.compiler import (
    CompiledStep,
    CompiledWorkflow,
    ValidationReport,
    compile_workflow,
    validate_workflow,
)
from m3resp.workflows.context import RESOLVED_OUTPUT_DIR_KEY, WorkflowContext
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.engine import (
    WorkflowResult,
    collect_diagnostics,
    run_spec,
    run_workflow,
    validate_spec,
)
from m3resp.workflows.lifecycle import (
    CancellationToken,
    CapturedWarning,
    ExecutionContext,
    StepExecutionRecord,
    WorkflowExecutionError,
)
from m3resp.workflows.registry import (
    STEP_ALIASES,
    STEP_REGISTRY,
    StepDefinition,
    available_steps,
    get_step,
    register_step,
)
from m3resp.workflows.service import WorkflowService, summarize_workflow_result
from m3resp.workflows.spec import StepSpec, WorkflowSpec, load_spec

__all__ = [
    "RESOLVED_OUTPUT_DIR_KEY",
    "STEP_ALIASES",
    "STEP_REGISTRY",
    "CancellationToken",
    "CapturedWarning",
    "CompiledStep",
    "CompiledWorkflow",
    "Diagnostic",
    "ExecutionContext",
    "StepDefinition",
    "StepExecutionRecord",
    "StepSpec",
    "ValidationReport",
    "WorkflowContext",
    "WorkflowExecutionError",
    "WorkflowResult",
    "WorkflowService",
    "WorkflowSpec",
    "available_steps",
    "collect_diagnostics",
    "compile_workflow",
    "get_step",
    "load_spec",
    "register_step",
    "run_spec",
    "run_workflow",
    "summarize_workflow_result",
    "validate_spec",
    "validate_workflow",
]
