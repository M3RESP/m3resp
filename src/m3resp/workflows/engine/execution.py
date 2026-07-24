"""Compiled-pipeline execution loop (`run_pipeline`)."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session
from m3resp.workflows.context import (
    RESOLVED_OUTPUT_DIR_KEY,
    SESSION_KEY,
    PipelineContext,
)
from m3resp.workflows.lifecycle import (
    CancellationToken,
    CapturedWarning,
    EventSink,
    PipelineExecutionError,
    PipelineStatus,
    StepExecutionRecord,
    build_execution_context,
    make_event,
    new_run_id,
    optional_package_version,
    summarize_output_value,
    utc_now_iso,
)
from m3resp.workflows.registry import (
    get_step,
)
from m3resp.workflows.spec import PipelineSpec, load_spec

if TYPE_CHECKING:
    # Deferred at runtime: compiler.py imports collect_diagnostics from this
    # package, so importing it back at module scope here would be circular.
    from m3resp.workflows.compiler import CompiledStep

from ._shared import PipelineResult, _ensure_steps_registered
from .diagnostics import collect_diagnostics


def run_pipeline(
    spec: str | Path | dict[str, Any] | PipelineSpec,
    *,
    session: M3Session | None = None,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
    extra_context: dict[str, Any] | None = None,
    event_sink: EventSink | None = None,
    cancellation_token: CancellationToken | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Run a declarative pipeline spec and return its result.

    ``extra_context`` seeds context keys produced outside the spec (e.g. signals
    already loaded onto the session), letting a processing-only spec begin from
    mid-pipeline artifacts. Those keys are treated as available during static
    validation.

    ``event_sink``, if given, receives one JSON-safe progress event per call
    (``pipeline_started``, ``step_started``, ``step_warning``,
    ``step_completed``, ``step_failed``, ``pipeline_completed``,
    ``pipeline_failed``, ``pipeline_cancelled``). ``cancellation_token`` is
    checked before and after each step; cancellation preserves
    already-completed work rather than rolling it back.

    A step function's own exception is re-raised wrapped in
    ``PipelineExecutionError``, with the original exception
    available as ``__cause__``.
    """

    _ensure_steps_registered()
    parsed = load_spec(spec)
    ctx = PipelineContext(
        session=session or M3Session(eit_adapter=eit_adapter, emg_adapter=emg_adapter),
        inputs=dict(parsed.inputs),
        root=parsed.root,
    )
    for key, value in (extra_context or {}).items():
        ctx.set(key, value)

    available = set(extra_context or ())
    diagnostics = tuple(collect_diagnostics(parsed, available=available))

    # Deferred import: compiler.py imports collect_diagnostics from this
    # module, so importing it at module scope here would be circular.
    from m3resp.workflows.compiler import compile_pipeline

    compiled = compile_pipeline(parsed, available=available)

    run_id = run_id or new_run_id()
    run_timestamp = utc_now_iso()
    execution_context = build_execution_context(
        run_id=run_id, run_timestamp=run_timestamp, seed=parsed.execution.seed
    )
    started_at = utc_now_iso()
    start_monotonic = time.monotonic()

    def emit(event_type: Any, **fields: Any) -> None:
        if event_sink is not None:
            event_sink(make_event(event_type, run_id=run_id, **fields))

    emit("pipeline_started", name=parsed.name, step_count=len(compiled.steps))

    step_records: list[StepExecutionRecord] = []
    status: PipelineStatus = "succeeded"

    for compiled_step in compiled.steps:
        if cancellation_token is not None and cancellation_token.cancelled:
            status = "cancelled"
            emit("pipeline_cancelled", name=parsed.name)
            break

        record = StepExecutionRecord(
            step_id=compiled_step.id,
            position=compiled_step.position,
            operation_id=compiled_step.operation_id,
            status="running",
            started_at=utc_now_iso(),
            parameters=dict(compiled_step.parameters),
            input_context_keys=dict(compiled_step.input_bindings),
            output_context_keys=dict(compiled_step.output_bindings),
            optional_package_versions={
                package: optional_package_version(package)
                for package in compiled_step.optional_packages
            },
        )
        emit(
            "step_started",
            step_id=record.step_id,
            position=record.position,
            operation_id=record.operation_id,
        )

        definition = get_step(compiled_step.operation_id)
        kwargs = _bind_compiled_arguments(compiled_step, ctx)
        step_start = time.monotonic()
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = definition.func(**kwargs) or {}
        except Exception as exc:
            # A warning issued right before the failure must still be
            # captured/re-emitted, not silently dropped - `caught` stays
            # bound and populated even though the `with` block raised.
            _replay_captured_warnings(caught, record, emit)
            record.status = "failed"
            record.finished_at = utc_now_iso()
            record.duration_seconds = time.monotonic() - step_start
            record.error = {"type": type(exc).__name__, "message": str(exc)}
            step_records.append(record)
            _record_processing_step(ctx, record)
            emit(
                "step_failed",
                step_id=record.step_id,
                position=record.position,
                error=record.error,
            )
            emit("pipeline_failed", name=parsed.name)
            raise PipelineExecutionError(
                step_id=compiled_step.id,
                position=compiled_step.position,
                operation_id=compiled_step.operation_id,
                message=str(exc),
                cause=exc,
                run_id=run_id,
                started_at=started_at,
                step_records=tuple(step_records),
            ) from exc

        _replay_captured_warnings(caught, record, emit)
        _store_compiled_outputs(compiled_step, ctx, result)

        record.status = "succeeded"
        record.finished_at = utc_now_iso()
        record.duration_seconds = time.monotonic() - step_start
        record.output_summaries = {
            name: summarize_output_value(result.get(name))
            for name in compiled_step.output_bindings
        }
        step_records.append(record)
        _record_processing_step(ctx, record)
        emit(
            "step_completed",
            step_id=record.step_id,
            position=record.position,
            duration_seconds=record.duration_seconds,
        )

        if cancellation_token is not None and cancellation_token.cancelled:
            status = "cancelled"
            emit("pipeline_cancelled", name=parsed.name)
            break

    finished_at = utc_now_iso()
    duration_seconds = time.monotonic() - start_monotonic
    if status == "succeeded":
        emit("pipeline_completed", name=parsed.name, duration_seconds=duration_seconds)

    produced = {
        key: ctx.get(key)
        for key in ctx.values
        if key != SESSION_KEY and key not in parsed.inputs
    }
    all_warnings = tuple(w for record in step_records for w in record.warnings)
    pipeline_result = PipelineResult(
        name=parsed.name,
        context=ctx,
        outputs=produced,
        run_id=run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        compiled_pipeline=compiled,
        step_records=tuple(step_records),
        diagnostics=diagnostics,
        warnings=all_warnings,
        execution_context=execution_context,
        resolved_output_dir=ctx.values.get(RESOLVED_OUTPUT_DIR_KEY),
    )
    if ctx.session.datamodel is not None:
        run = ctx.session.datamodel.record_pipeline_result(pipeline_result)
        pipeline_result.processing_run_id = run.processing_run_id
    return pipeline_result


def _replay_captured_warnings(
    caught: list[warnings.WarningMessage],
    record: StepExecutionRecord,
    emit: Any,
) -> None:
    """Store each captured warning on the step's record, emit a
    ``step_warning`` event, and re-raise it through the normal warnings
    machinery so a caller's own filters/``-W error``/
    ``pytest.warns`` still see it exactly once - it was only ever
    suppressed from display while captured, never dropped."""

    for warning in caught:
        captured = CapturedWarning(
            message=str(warning.message), category=warning.category.__name__
        )
        record.warnings.append(captured)
        emit(
            "step_warning",
            step_id=record.step_id,
            position=record.position,
            message=captured.message,
            category=captured.category,
        )
        warnings.warn_explicit(
            warning.message, warning.category, warning.filename, warning.lineno
        )
    caught.clear()


def _record_processing_step(ctx: PipelineContext, record: StepExecutionRecord) -> None:
    """log every executed step onto the session's universal
    ``ProcessingHistory``, using exactly what the engine already knows
    (bindings/parameters/timing/outcome) - no step function needs to call
    anything itself. Distinct from the datamodel's per-*pipeline*
    ``ProcessingRun`` (see ``DataModelRecorder.record_pipeline_result``), so
    this never creates a duplicate/competing ``ProcessingRun``."""

    ctx.session.processing_history.record(
        record.operation_id,
        input_keys=list(record.input_context_keys.values()),
        output_keys=list(record.output_context_keys.values()),
        parameters=dict(record.parameters),
        timestamp=record.finished_at or record.started_at or utc_now_iso(),
        status=record.status,
    )


def _bind_compiled_arguments(
    compiled_step: CompiledStep, ctx: PipelineContext
) -> dict[str, Any]:
    """Build a step's call kwargs from an already-compiled step:
    context reads resolve against the live context; static parameters were
    already fully resolved (``@ref``s and paths) at compile time."""

    kwargs: dict[str, Any] = {
        param: ctx.get(context_key)
        for param, context_key in compiled_step.input_bindings.items()
    }
    kwargs.update(compiled_step.parameters)
    return kwargs


def _store_compiled_outputs(
    compiled_step: CompiledStep, ctx: PipelineContext, result: Any
) -> None:
    if not isinstance(result, dict):
        raise PipelineSpecError(
            f"Step #{compiled_step.position} '{compiled_step.operation_id}' must "
            f"return a mapping of outputs or None, got {type(result).__name__}."
        )
    for name, context_key in compiled_step.output_bindings.items():
        if name not in result:
            raise PipelineSpecError(
                f"Step #{compiled_step.position} '{compiled_step.operation_id}' "
                f"declared output '{name}' but did not return it."
            )
        ctx.set(context_key, result[name])
