"""Resolve workflow operations, input/output names and processing settings.

Compilation describes steps in execution order, resolving input references
and paths relative to the spec root. Validation reports structural problems
and, when requested, package availability and missing input files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.exceptions import UnknownStepError, WorkflowSpecError
from m3resp.core.path_helper import resolve_optional_path
from m3resp.workflows.context import resolve_value
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.engine import collect_diagnostics
from m3resp.workflows.registry import (
    StepArtifact,
    StepDefinition,
    get_step,
    step_capability_state,
)
from m3resp.workflows.spec import StepSpec, WorkflowSpec


@dataclass(frozen=True)
class CompiledStep:
    """One fully-resolved step, ready to execute or to describe to a GUI."""

    id: str
    position: int
    operation_id: str
    summary: str
    #: step function parameter name -> context key it reads from.
    input_bindings: dict[str, str] = field(default_factory=dict)
    #: parameter names in ``input_bindings`` whose context key may be absent
    #: at run time; the step is then called without that argument.
    optional_bindings: frozenset[str] = frozenset()
    #: step's natural output name -> context key it is stored under.
    output_bindings: dict[str, str] = field(default_factory=dict)
    #: static parameter name -> recursively resolved value (``@ref``
    #: substituted, path parameters resolved against the spec root, and
    #: unset optional parameters filled with their declared default).
    parameters: dict[str, Any] = field(default_factory=dict)
    input_artifacts: tuple[StepArtifact, ...] = ()
    output_artifacts: tuple[StepArtifact, ...] = ()
    optional_packages: tuple[str, ...] = ()
    modality: str | None = None
    category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position,
            "operation_id": self.operation_id,
            "summary": self.summary,
            "input_bindings": dict(self.input_bindings),
            "output_bindings": dict(self.output_bindings),
            "parameters": dict(self.parameters),
            "input_artifacts": [a.as_dict() for a in self.input_artifacts],
            "output_artifacts": [a.as_dict() for a in self.output_artifacts],
            "optional_packages": list(self.optional_packages),
            "modality": self.modality,
            "category": self.category,
        }


@dataclass(frozen=True)
class CompiledWorkflow:
    """An ordered description of workflow steps with resolved settings.

    Attributes:
        name: Workflow name copied from the spec.
        schema_version: Spec format version, or None for an unversioned spec.
        steps: Compiled steps in execution order.
    """

    name: str
    schema_version: int | None
    steps: tuple[CompiledStep, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the workflow name, format version and step descriptions as a dict."""

        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "steps": [step.as_dict() for step in self.steps],
        }


def compile_workflow(
    spec: WorkflowSpec, *, available: set[str] | None = None
) -> CompiledWorkflow:
    """Validate a workflow and resolve the settings needed to run its steps.

    Args:
        spec: Parsed workflow with ordered steps and a path-resolution root.
        available: Extra context keys supplied by the caller before execution.

    Returns:
        CompiledWorkflow: Steps in spec order with operation names, input/output
            bindings, defaults and settings resolved. ``@name`` references use
            spec.inputs; path settings are resolved relative to spec.root.

    Raises:
        UnknownStepError: If the first structural error names an unknown step.
        WorkflowSpecError: If the first structural error concerns bindings,
            parameters or other workflow structure.
    """

    diagnostics = collect_diagnostics(spec, available=available)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        first = errors[0]
        if first.code == "unknown_step":
            raise UnknownStepError(first.message)
        raise WorkflowSpecError(first.message)

    compiled_steps = tuple(
        _compile_step(step_spec, get_step(step_spec.uses), spec, position)
        for position, step_spec in enumerate(spec.steps)
    )
    return CompiledWorkflow(
        name=spec.name, schema_version=spec.schema_version, steps=compiled_steps
    )


def _compile_step(
    step_spec: StepSpec,
    definition: StepDefinition,
    spec: WorkflowSpec,
    position: int,
) -> CompiledStep:
    """Resolve one step's input/output names, defaults, references and paths."""

    input_bindings: dict[str, str] = {}
    optional_bindings: set[str] = set()
    for param, default in definition.reads.items():
        context_key = step_spec.inputs.get(param, default)
        # collect_diagnostics already rejected a None (unbound) context key
        # above, so every required read is guaranteed bound by the time we get
        # here. An optional read may still be unbound, and is simply not passed.
        if context_key is None:
            assert param in definition.optional_reads
            continue
        input_bindings[param] = context_key
        if param in definition.optional_reads:
            optional_bindings.add(param)
    output_bindings = {
        name: step_spec.outputs.get(name, name) for name in definition.writes
    }

    path_param_names = {p.name for p in definition.parameters if p.value_type == "path"}

    def _resolve(name: str, raw_value: Any) -> Any:
        value = resolve_value(raw_value, spec.inputs)
        if name in path_param_names and isinstance(value, str):
            value = str(resolve_optional_path(spec.root, value))
        return value

    resolved_parameters: dict[str, Any] = {}
    for parameter in definition.parameters:
        if parameter.name in step_spec.params:
            resolved_parameters[parameter.name] = _resolve(
                parameter.name, step_spec.params[parameter.name]
            )
        else:
            # A required-and-missing parameter would already have failed
            # collect_diagnostics above, so reaching this branch means
            # either optional-with-a-default or optional-and-unset (None).
            resolved_parameters[parameter.name] = parameter.default
    # Resolve supplied settings that have no declared parameter metadata.
    for name, raw_value in step_spec.params.items():
        if name not in resolved_parameters:
            resolved_parameters[name] = _resolve(name, raw_value)

    return CompiledStep(
        id=step_spec.id,
        position=position,
        operation_id=definition.operation_id,
        summary=definition.summary,
        input_bindings=input_bindings,
        optional_bindings=frozenset(optional_bindings),
        output_bindings=output_bindings,
        parameters=resolved_parameters,
        input_artifacts=definition.input_artifacts,
        output_artifacts=definition.output_artifacts,
        optional_packages=definition.optional_packages,
        modality=definition.modality,
        category=definition.category,
    )


@dataclass(frozen=True)
class ValidationReport:
    """Structural diagnostics and optional checks for running a workflow locally."""

    structural: tuple[Diagnostic, ...] = ()
    readiness: tuple[Diagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether the structural diagnostics contain no errors.

        Readiness diagnostics are assessed separately.
        """

        return not any(d.severity == "error" for d in self.structural)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "structural": [d.as_dict() for d in self.structural],
            "readiness": [d.as_dict() for d in self.readiness],
        }


def validate_workflow(
    spec: WorkflowSpec,
    *,
    available: set[str] | None = None,
    readiness: bool = False,
) -> ValidationReport:
    """Check workflow structure and optionally its readiness on this machine.

    Args:
        spec: Parsed workflow to inspect.
        available: Extra context keys supplied by the caller before execution.
        readiness: Also check optional package availability and file paths.

    Returns:
        ValidationReport: Structural errors and warnings, plus readiness
            diagnostics when requested. Missing packages give warnings and
            missing input files give errors in the readiness section.
            is_valid reflects structural errors only.
    """

    structural = collect_diagnostics(spec, available=available)
    readiness_diagnostics: list[Diagnostic] = []
    if readiness:
        for position, step_spec in enumerate(spec.steps):
            try:
                definition = get_step(step_spec.uses)
            except UnknownStepError:
                continue  # already reported structurally
            readiness_diagnostics.extend(
                _readiness_diagnostics_for_step(step_spec, definition, spec, position)
            )
    return ValidationReport(
        structural=tuple(structural), readiness=tuple(readiness_diagnostics)
    )


def _readiness_diagnostics_for_step(
    step_spec: StepSpec,
    definition: StepDefinition,
    spec: WorkflowSpec,
    position: int,
) -> list[Diagnostic]:
    """Report one step's package availability and missing file-path parameters."""

    diagnostics: list[Diagnostic] = []
    step_label = f"step #{position} '{step_spec.uses}'"

    capability = step_capability_state(step_spec.uses)
    if capability != "available":
        diagnostics.append(
            Diagnostic(
                severity="warning"
                if capability == "missing_optional_dependency"
                else "info",
                code=f"capability_{capability}",
                message=f"{step_label} capability: {capability}.",
                step_id=step_spec.id,
                step_position=position,
                operation_id=step_spec.uses,
                suggestion=(
                    f"Install one of: {list(definition.optional_packages)}."
                    if capability == "missing_optional_dependency"
                    else None
                ),
            )
        )

    for parameter in definition.parameters:
        if parameter.value_type != "path" or parameter.path_kind != "file":
            continue
        raw_value = step_spec.params.get(parameter.name)
        if raw_value is None:
            continue
        try:
            value = resolve_value(raw_value, spec.inputs)
        except WorkflowSpecError:
            continue  # already reported structurally
        if not isinstance(value, str):
            continue
        resolved_path = resolve_optional_path(spec.root, value)
        if resolved_path is not None and not resolved_path.exists():
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="missing_file",
                    message=(
                        f"{step_label} parameter '{parameter.name}' file does not "
                        f"exist: {resolved_path}."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )

    return diagnostics
