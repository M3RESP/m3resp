"""Spec validation and compile-time diagnostics (`validate_spec`, `collect_diagnostics`)."""

from __future__ import annotations

from typing import Any

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.workflows.context import (
    RESOLVED_OUTPUT_DIR_KEY,
    SESSION_KEY,
    iter_input_references,
    resolve_value,
)
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepDefinition,
    StepParameter,
    get_step,
)
from m3resp.workflows.session_deps import find_session_dependency_conflicts
from m3resp.workflows.spec import PipelineSpec, StepSpec

from ._shared import _ensure_steps_registered


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_number_or_mapping_of_numbers(value: Any) -> bool:
    """A ``number`` parameter also accepts a ``{key: number}`` mapping - the
    recurring "single value, or a per-key override" pattern (e.g.
    ``session.sync_raw``'s ``offset_seconds``: one offset, or one per
    modality)."""

    if isinstance(value, dict):
        return all(_is_number(v) for v in value.values())
    return _is_number(value)


_PARAM_TYPE_CHECKS: dict[str, Any] = {
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _is_number_or_mapping_of_numbers,
    "string": lambda v: isinstance(v, str),
    "path": lambda v: isinstance(v, str),
    "mapping": lambda v: isinstance(v, dict),
    "list": lambda v: isinstance(v, list),
}


_RESERVED_ENGINE_KEYS = frozenset(
    (
        SESSION_KEY,
        "_spec_outputs",
        "_spec_experiment",
        RESOLVED_OUTPUT_DIR_KEY,
        "_run_timestamp",
    )
)


def validate_spec(spec: PipelineSpec, *, available: set[str] | None = None) -> None:
    """Statically check that every step's inputs are produced before use,
    that no two steps write to the same context key without explicit renaming,
    and that every ``@name`` input reference names a declared pipeline input.

    Compatibility wrapper around :func:`collect_diagnostics`:
    raises ``PipelineSpecError`` using the first error-severity diagnostic's
    message when any exist. Call :func:`collect_diagnostics` directly to get
    every independent problem in one pass instead of only the first.

    ``available`` lists extra context keys seeded outside the spec.
    """

    diagnostics = collect_diagnostics(spec, available=available)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        first = errors[0]
        if first.code == "unknown_step":
            raise UnknownStepError(first.message)
        raise PipelineSpecError(first.message)


def collect_diagnostics(
    spec: PipelineSpec, *, available: set[str] | None = None
) -> list[Diagnostic]:
    """Return every independent structural problem in ``spec``.

    Does not raise. Unlike ``validate_spec()``, this reports every violation
    found in one pass rather than stopping at the first, and returns
    JSON-safe :class:`Diagnostic` objects instead of exception messages.
    ``available`` lists extra context keys seeded outside the spec.
    """

    _ensure_steps_registered()
    diagnostics: list[Diagnostic] = []
    # _spec_outputs, _spec_experiment, _resolved_output_dir, and _run_timestamp
    # are always injected by run_spec before the pipeline executes, so treat
    # them as globally available. Pre-seeded keys are exempt from
    # duplicate-write detection.
    seeded: set[str] = {
        SESSION_KEY,
        "_spec_outputs",
        "_spec_experiment",
        RESOLVED_OUTPUT_DIR_KEY,
        "_run_timestamp",
        *spec.inputs,
        *(available or set()),
    }
    # Maps context key -> label of the step that produced it, for messages.
    produced: dict[str, str] = {key: "pipeline seed" for key in seeded}
    # Maps context key -> declared StepArtifact.artifact_type of whatever
    # produced it, for the artifact-type compatibility check below. Only
    # populated where a producer actually declares one (additive metadata),
    # so an undeclared key is simply skipped, never flagged.
    produced_artifact_type: dict[str, str] = {SESSION_KEY: "m3session"}

    for position, step_spec in enumerate(spec.steps):
        step_label = f"step #{position} '{step_spec.uses}'"

        try:
            definition = get_step(step_spec.uses)
        except UnknownStepError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unknown_step",
                    message=str(exc),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
            continue  # cannot check bindings without a definition

        diagnostics.extend(
            _check_bindings(
                step_spec,
                definition,
                position,
                step_label,
                produced,
                produced_artifact_type,
            )
        )
        diagnostics.extend(
            _check_static_parameters(
                step_spec, definition, spec.inputs, position, step_label
            )
        )
        diagnostics.extend(
            _check_mutually_exclusive_parameters(
                step_spec, definition, position, step_label
            )
        )

        output_artifacts_by_name = {a.name: a for a in definition.output_artifacts}
        for name in definition.writes:
            context_key = step_spec.outputs.get(name, name)
            if context_key in produced and context_key not in seeded:
                diagnostics.append(
                    Diagnostic(
                        severity="error",
                        code="duplicate_output",
                        message=(
                            f"{step_label} writes to context key '{context_key}', "
                            f"which was already produced by {produced[context_key]}. "
                            "Use 'out:' to rename one of the outputs."
                        ),
                        step_id=step_spec.id,
                        step_position=position,
                        operation_id=step_spec.uses,
                    )
                )
            produced[context_key] = step_label
            artifact = output_artifacts_by_name.get(name)
            if artifact is not None:
                produced_artifact_type[context_key] = artifact.artifact_type

    diagnostics.extend(_check_session_dependencies(spec))

    return diagnostics


def _check_session_dependencies(spec: PipelineSpec) -> list[Diagnostic]:
    """A step reading a declared ``session_reads`` resource before any step
    that (later in the same spec) declares writing it usually means the
    spec's step order silently reordered a session-mediated dependency -
    see ``m3resp.workflows.session_deps``. Reported as a warning, not an
    error: the resource may also come from state supplied outside the
    spec, which this check cannot see and is not a bug."""

    diagnostics: list[Diagnostic] = []
    for conflict in find_session_dependency_conflicts(spec):
        reader_label = f"step #{conflict.reader_position} '{conflict.reader_uses}'"
        writer_label = f"step #{conflict.writer_position} '{conflict.writer_uses}'"
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code="session_dependency_reordered",
                message=(
                    f"{reader_label} reads session resource "
                    f"'{conflict.resource}' before it is written by "
                    f"{writer_label}, which runs later in this spec. If "
                    f"{reader_label} depends on {writer_label}'s output, "
                    "move it after; if it depends on state supplied "
                    "outside this spec, this warning can be ignored."
                ),
                step_id=conflict.reader_step_id,
                step_position=conflict.reader_position,
                operation_id=conflict.reader_uses,
            )
        )
    return diagnostics


def _check_bindings(
    step_spec: StepSpec,
    definition: StepDefinition,
    position: int,
    step_label: str,
    produced: dict[str, str],
    produced_artifact_type: dict[str, str],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    input_artifacts_by_name = {a.name: a for a in definition.input_artifacts}

    for param in step_spec.inputs:
        if param not in definition.reads:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unknown_input_binding",
                    message=(
                        f"{step_label} binds unknown parameter '{param}' via 'in:'."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                    suggestion=f"Known 'in:' parameters: {sorted(definition.reads)}",
                )
            )

    for name in step_spec.outputs:
        if name not in definition.writes:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unknown_output_binding",
                    message=(
                        f"{step_label} renames unknown output '{name}' via 'out:'."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                    suggestion=f"Declared outputs: {list(definition.writes)}",
                )
            )

    for name in definition.writes:
        output_context_key = step_spec.outputs.get(name, name)
        if output_context_key in _RESERVED_ENGINE_KEYS:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="reserved_context_key_overwrite",
                    message=(
                        f"{step_label} writes output '{name}' to reserved "
                        f"context key '{output_context_key}', which the engine "
                        "seeds before any step runs. Rename this output via "
                        "'out:' to a different context key."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )

    for param in sorted(set(step_spec.inputs) & set(step_spec.params)):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="duplicate_binding",
                message=(
                    f"{step_label} binds parameter '{param}' through both "
                    "'in' and 'with'."
                ),
                step_id=step_spec.id,
                step_position=position,
                operation_id=step_spec.uses,
            )
        )

    for param, default in definition.reads.items():
        context_key = step_spec.inputs.get(param, default)
        if context_key is None:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="missing_binding",
                    message=(
                        f"{step_label} requires an explicit 'in: {{{param}: ...}}' "
                        "binding — this parameter has no default context key."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
        elif context_key not in produced:
            if param in definition.optional_reads:
                # Absent is allowed: the step runs without the argument.
                continue
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unproduced_context_key",
                    message=(
                        f"{step_label} reads context key '{context_key}', which is "
                        "not produced by an earlier step or declared in inputs."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
        else:
            diagnostics.extend(
                _check_artifact_type_compatibility(
                    param,
                    context_key,
                    input_artifacts_by_name,
                    produced_artifact_type,
                    step_spec,
                    position,
                    step_label,
                )
            )
    for context_key in definition.requires:
        if context_key not in produced:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unproduced_context_key",
                    message=(
                        f"{step_label} reads context key '{context_key}', which is "
                        "not produced by an earlier step or declared in inputs."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )

    for parameter in definition.parameters:
        if parameter.required and parameter.name not in step_spec.params:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="missing_required_parameter",
                    message=(
                        f"{step_label} is missing required parameter "
                        f"'{parameter.name}' in 'with:'."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )

    return diagnostics


def _check_artifact_type_compatibility(
    param: str,
    context_key: str,
    input_artifacts_by_name: dict[str, StepArtifact],
    produced_artifact_type: dict[str, str],
    step_spec: StepSpec,
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    """a step's declared input artifact type must match whatever
    produced the context key it's bound to ("compilation validates
    ... artifact compatibility"), unless either side
    declares :data:`ANY_ARTIFACT_TYPE` (a genuine passthrough, e.g.
    ``eit.slice``). Only checked when *both* sides declare a type - this is
    additive metadata, backfilled module by module, so an undeclared type on
    either side is simply skipped rather than flagged."""

    consumer_artifact = input_artifacts_by_name.get(param)
    if consumer_artifact is None:
        return []
    consumer_type = consumer_artifact.artifact_type
    producer_type = produced_artifact_type.get(context_key)
    if producer_type is None:
        return []
    if ANY_ARTIFACT_TYPE in (consumer_type, producer_type):
        return []
    if consumer_type == producer_type:
        return []
    return [
        Diagnostic(
            severity="error",
            code="artifact_type_mismatch",
            message=(
                f"{step_label} parameter '{param}' expects artifact type "
                f"'{consumer_type}', but context key '{context_key}' was "
                f"produced with artifact type '{producer_type}'."
            ),
            step_id=step_spec.id,
            step_position=position,
            operation_id=step_spec.uses,
        )
    ]


def _check_mutually_exclusive_parameters(
    step_spec: StepSpec,
    definition: StepDefinition,
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    """reject a spec that sets more than one parameter from a
    declared mutually-exclusive group (e.g. ``emg.ecg_gating``'s
    ``gate_width_seconds``/``gate_width_samples``) in the same step
    invocation - a structural, compile-time version of what would otherwise
    only surface as a runtime ``ValueError`` once the step actually executes.
    "Set" means present as a key in ``with:``, regardless of its resolved
    value - an unset parameter is never passed at all."""

    diagnostics: list[Diagnostic] = []
    for group in definition.mutually_exclusive_parameters:
        set_names = [name for name in group if name in step_spec.params]
        if len(set_names) > 1:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="mutually_exclusive_parameters_conflict",
                    message=(
                        f"{step_label} sets more than one of mutually-exclusive "
                        f"parameters {list(group)} in 'with:': {set_names}. Set "
                        "at most one."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
    return diagnostics


def _check_static_parameters(
    step_spec: StepSpec,
    definition: StepDefinition,
    spec_inputs: dict[str, Any],
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    """validate static parameter values against declared metadata.

    Only checked when the step declares metadata for that parameter name.
    ``@name`` references are resolved against ``spec_inputs`` first (an unknown
    reference is already reported by ``_check_reference``, so it is skipped
    here rather than reported twice).
    """

    diagnostics: list[Diagnostic] = []
    parameters_by_name = {p.name: p for p in definition.parameters}

    for param_name, raw_value in step_spec.params.items():
        diagnostics.extend(
            _check_references(raw_value, spec_inputs, step_spec, position, step_label)
        )

        parameter = parameters_by_name.get(param_name)
        if parameter is None:
            continue
        try:
            value = resolve_value(raw_value, spec_inputs)
        except PipelineSpecError:
            continue  # already reported by _check_references

        if value is None:
            continue  # None means "unset"; required-ness is checked separately
        diagnostics.extend(
            _check_parameter_value(parameter, value, step_spec, position, step_label)
        )

    return diagnostics


def _check_references(
    raw_value: Any,
    spec_inputs: dict[str, Any],
    step_spec: StepSpec,
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for ref in iter_input_references(raw_value):
        if ref not in spec_inputs:
            available_inputs = ", ".join(sorted(spec_inputs)) or "(none)"
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unknown_input_reference",
                    message=(
                        f"{step_label} references unknown input '@{ref}'. "
                        f"Declared inputs: {available_inputs}."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
    return diagnostics


def _check_parameter_value(
    parameter: StepParameter,
    value: Any,
    step_spec: StepSpec,
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    checker = _PARAM_TYPE_CHECKS.get(parameter.value_type)
    if checker is not None and not checker(value):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="parameter_type_mismatch",
                message=(
                    f"{step_label} parameter '{parameter.name}' must be of type "
                    f"'{parameter.value_type}'; got {type(value).__name__} "
                    f"({value!r})."
                ),
                step_id=step_spec.id,
                step_position=position,
                operation_id=step_spec.uses,
            )
        )
        return diagnostics  # range/choice checks assume a well-typed value

    if parameter.choices is not None and value not in parameter.choices:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="parameter_choice_violation",
                message=(
                    f"{step_label} parameter '{parameter.name}' value {value!r} "
                    f"is not one of {list(parameter.choices)}."
                ),
                step_id=step_spec.id,
                step_position=position,
                operation_id=step_spec.uses,
            )
        )
    if isinstance(value, int | float) and not isinstance(value, bool):
        if parameter.minimum is not None and value < parameter.minimum:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="parameter_range_violation",
                    message=(
                        f"{step_label} parameter '{parameter.name}' value {value!r} "
                        f"is below its minimum {parameter.minimum!r}."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
        if parameter.maximum is not None and value > parameter.maximum:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="parameter_range_violation",
                    message=(
                        f"{step_label} parameter '{parameter.name}' value {value!r} "
                        f"is above its maximum {parameter.maximum!r}."
                    ),
                    step_id=step_spec.id,
                    step_position=position,
                    operation_id=step_spec.uses,
                )
            )
    return diagnostics
