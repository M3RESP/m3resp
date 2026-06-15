"""Execution engine for declarative M3Resp pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session
from m3resp.pipeline.context import SESSION_KEY, PipelineContext
from m3resp.pipeline.registry import StepDefinition, get_step
from m3resp.pipeline.spec import PipelineSpec, StepSpec, load_spec

_STEPS_REGISTERED = False


def _ensure_steps_registered() -> None:
    """Import the built-in step package so every step is registered.

    Done lazily (not at module import) to avoid a circular import: the step
    modules import ``m3resp.workflows`` helpers, and the workflows package imports
    the pipeline engine. Step modules defer their upstream (eitprocessing/
    resurfemg) imports to call time, so this stays safe without those packages.
    """

    global _STEPS_REGISTERED
    if not _STEPS_REGISTERED:
        from m3resp.pipeline import steps as _steps  # noqa: F401

        _STEPS_REGISTERED = True


@dataclass
class PipelineResult:
    """Outcome of running a pipeline."""

    name: str
    context: PipelineContext
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def session(self) -> M3Session:
        return self.context.session

    def value(self, key: str) -> Any:
        """Return a produced artifact by context key."""

        return self.context.get(key)


def run_pipeline(
    spec: str | Path | dict[str, Any] | PipelineSpec,
    *,
    session: M3Session | None = None,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
    extra_context: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run a declarative pipeline spec and return its result.

    ``extra_context`` seeds context keys produced outside the spec (e.g. signals
    already loaded onto the session), letting a processing-only spec begin from
    mid-pipeline artifacts. Those keys are treated as available during static
    validation.
    """

    _ensure_steps_registered()
    parsed = load_spec(spec)
    ctx = PipelineContext(
        session=session or M3Session(eit_adapter=eit_adapter, emg_adapter=emg_adapter),
        inputs=dict(parsed.inputs),
    )
    for key, value in (extra_context or {}).items():
        ctx.set(key, value)

    validate_spec(parsed, available=set(extra_context or ()))

    for position, step_spec in enumerate(parsed.steps):
        definition = get_step(step_spec.uses)
        kwargs = _bind_arguments(definition, step_spec, ctx, position)
        result = definition.func(**kwargs) or {}
        _store_outputs(definition, step_spec, ctx, result, position)

    produced = {
        key: ctx.get(key)
        for key in ctx.values
        if key != SESSION_KEY and key not in parsed.inputs
    }
    return PipelineResult(name=parsed.name, context=ctx, outputs=produced)


def validate_spec(spec: PipelineSpec, *, available: set[str] | None = None) -> None:
    """Statically check that every step's inputs are produced before use.

    ``available`` lists extra context keys seeded outside the spec.
    """

    _ensure_steps_registered()
    produced: set[str] = {SESSION_KEY, *spec.inputs, *(available or set())}
    for position, step_spec in enumerate(spec.steps):
        definition = get_step(step_spec.uses)
        for context_key in _required_context_keys(definition, step_spec):
            if context_key not in produced:
                raise PipelineSpecError(
                    f"Step #{position} '{step_spec.uses}' reads context key "
                    f"'{context_key}', which is not produced by an earlier step "
                    f"or declared in inputs."
                )
        produced.update(_output_context_keys(definition, step_spec))


def _required_context_keys(definition: StepDefinition, step_spec: StepSpec) -> set[str]:
    keys = {
        step_spec.inputs.get(param, default)
        for param, default in definition.reads.items()
    }
    keys.update(definition.requires)
    return keys


def _output_context_keys(definition: StepDefinition, step_spec: StepSpec) -> set[str]:
    return {step_spec.outputs.get(name, name) for name in definition.writes}


def _bind_arguments(
    definition: StepDefinition,
    step_spec: StepSpec,
    ctx: PipelineContext,
    position: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for param, default_key in definition.reads.items():
        context_key = step_spec.inputs.get(param, default_key)
        kwargs[param] = ctx.get(context_key)

    for param, raw_value in step_spec.params.items():
        if param in kwargs:
            raise PipelineSpecError(
                f"Step #{position} '{step_spec.uses}' binds parameter '{param}' "
                f"through both 'in' and 'with'."
            )
        kwargs[param] = ctx.resolve_input(raw_value)
    return kwargs


def _store_outputs(
    definition: StepDefinition,
    step_spec: StepSpec,
    ctx: PipelineContext,
    result: Any,
    position: int,
) -> None:
    if not isinstance(result, dict):
        raise PipelineSpecError(
            f"Step #{position} '{step_spec.uses}' must return a mapping of outputs "
            f"or None, got {type(result).__name__}."
        )
    for name in definition.writes:
        if name not in result:
            raise PipelineSpecError(
                f"Step #{position} '{step_spec.uses}' declared output '{name}' "
                f"but did not return it."
            )
        ctx.set(step_spec.outputs.get(name, name), result[name])
