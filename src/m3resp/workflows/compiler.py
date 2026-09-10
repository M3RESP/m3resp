"""Compile a validated spec into a read-only execution plan (Phase 3.1 of
the pipeline-structure plan), and provide structural-vs-readiness validation
reports (Phase 3.5) on top of the diagnostics collected in ``engine.py``.

Compilation never imports optional scientific packages, opens data files,
creates output directories, or mutates a session - it only resolves names,
context bindings, and static parameter values (including ``@ref``
substitution and spec-root path resolution) that are already fully knowable
from the spec and the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
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
from m3resp.workflows.spec import PipelineSpec, StepSpec


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
class CompiledPipeline:
    """A read-only, ordered execution plan. What provenance and a future GUI
    display - not the raw spec, which may still contain unresolved ``@ref``
    values and cwd-relative-looking path strings."""

    name: str
    schema_version: int | None
    steps: tuple[CompiledStep, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "steps": [step.as_dict() for step in self.steps],
        }


def compile_pipeline(
    spec: PipelineSpec, *, available: set[str] | None = None
) -> CompiledPipeline:
    """Compile ``spec`` into a read-only, fully-resolved execution plan.

    Raises ``PipelineSpecError``/``UnknownStepError`` for the first
    structural problem found - call :func:`validate_pipeline` first to get
    every independent problem in one pass instead of just the first.
    """

    diagnostics = collect_diagnostics(spec, available=available)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        first = errors[0]
        if first.code == "unknown_step":
            raise UnknownStepError(first.message)
        raise PipelineSpecError(first.message)

    compiled_steps = tuple(
        _compile_step(step_spec, get_step(step_spec.uses), spec, position)
        for position, step_spec in enumerate(spec.steps)
    )
    return CompiledPipeline(
        name=spec.name, schema_version=spec.schema_version, steps=compiled_steps
    )


def _compile_step(
    step_spec: StepSpec,
    definition: StepDefinition,
    spec: PipelineSpec,
    position: int,
) -> CompiledStep:
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
    # Static parameters without declared metadata still need to reach the
    # function - pass them through resolved but unvalidated (full metadata
    # coverage is Phase 8.4, not this phase).
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
    """Phase 3.5: structural validation is always run; readiness (optional
    packages, file existence) is opt-in, since it needs the local machine's
    installed packages and filesystem, not just the spec."""

    structural: tuple[Diagnostic, ...] = ()
    readiness: tuple[Diagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether the spec is structurally sound and could compile - not
        whether it's ready to run on this machine (see ``readiness``)."""

        return not any(d.severity == "error" for d in self.structural)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "structural": [d.as_dict() for d in self.structural],
            "readiness": [d.as_dict() for d in self.readiness],
        }


def validate_pipeline(
    spec: PipelineSpec,
    *,
    available: set[str] | None = None,
    readiness: bool = False,
) -> ValidationReport:
    """Return a full structural (and, if requested, readiness) report.

    Structural validation never imports optional packages or touches the
    filesystem, so a GUI can validate a pipeline on a machine without the
    optional backend installed. Readiness additionally reports missing
    optional dependencies and missing input files - set ``readiness=True``
    when you also want to know whether the spec can actually run *here*.
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
    spec: PipelineSpec,
    position: int,
) -> list[Diagnostic]:
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
        except PipelineSpecError:
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
