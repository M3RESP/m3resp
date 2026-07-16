"""Execution engine for declarative M3Resp pipelines."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.core.session import M3Session
from m3resp.workflows.context import (
    RESOLVED_OUTPUT_DIR_KEY,
    SESSION_KEY,
    PipelineContext,
    iter_input_references,
    resolve_value,
)
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.lifecycle import (
    CancellationToken,
    CapturedWarning,
    EventSink,
    ExecutionContext,
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
from m3resp.workflows.registry import StepDefinition, StepParameter, get_step
from m3resp.workflows.spec import PipelineSpec, StepSpec, load_spec

if TYPE_CHECKING:
    # Deferred at runtime: compiler.py imports collect_diagnostics from this
    # module, so importing it back at module scope here would be circular.
    from m3resp.workflows.compiler import CompiledPipeline, CompiledStep


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


#: Parameter value-type checkers for Phase 3.3 static validation. ``choice``
#: is checked separately via its ``choices`` tuple, not by Python type.
_PARAM_TYPE_CHECKS: dict[str, Any] = {
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _is_number_or_mapping_of_numbers,
    "string": lambda v: isinstance(v, str),
    "path": lambda v: isinstance(v, str),
    "mapping": lambda v: isinstance(v, dict),
    "list": lambda v: isinstance(v, list),
}

_STEPS_REGISTERED = False


def _ensure_steps_registered() -> None:
    """Import the built-in step package so every step is registered.

    Done lazily to avoid a circular import between the step modules and the
    workflows package.
    """

    global _STEPS_REGISTERED
    if not _STEPS_REGISTERED:
        from m3resp.workflows import steps as _steps  # noqa: F401

        _STEPS_REGISTERED = True


@dataclass
class PipelineResult:
    """Outcome of running a pipeline."""

    name: str
    context: PipelineContext
    outputs: dict[str, Any] = field(default_factory=dict)
    #: The `ProcessingRun` id `record_pipeline_result` created for this run,
    #: when a `DataModelRecorder` is attached to the session. `None`
    #: otherwise (including for a session without a recorder).
    processing_run_id: str | None = None
    #: Phase 4.7 additions. All additive; existing fields above are unchanged.
    run_id: str | None = None
    status: PipelineStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    compiled_pipeline: "CompiledPipeline | None" = None
    step_records: tuple[StepExecutionRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    warnings: tuple[CapturedWarning, ...] = ()
    execution_context: ExecutionContext | None = None
    resolved_output_dir: Path | None = None
    #: Populated once Phase 6 writes a run manifest; `None` until then.
    manifest_path: Path | None = None

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
    (Phase 4.4: ``pipeline_started``, ``step_started``, ``step_warning``,
    ``step_completed``, ``step_failed``, ``pipeline_completed``,
    ``pipeline_failed``, ``pipeline_cancelled``). ``cancellation_token`` is
    checked before and after each step (Phase 4.5); cancellation preserves
    already-completed work rather than rolling it back.

    A step function's own exception is re-raised wrapped in
    ``PipelineExecutionError`` (Phase 4.2), with the original exception
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


def _resolve_output_mode(spec: PipelineSpec) -> tuple[str, bool]:
    """Phase 6.2: replaces the old "any of three hardcoded export step names
    present" heuristic with an explicit ``outputs.mode``.

    Returns ``(mode, was_inferred)``. A versioned spec always states its
    mode when ``outputs.dir`` is set (enforced at parse time), so
    ``was_inferred`` is only ever ``True`` for a legacy spec that omitted
    ``outputs.mode``; the inference itself now checks for *any* step under
    the ``export.*`` prefix, not just the three names the old heuristic knew.
    """

    if spec.outputs.mode is not None:
        return spec.outputs.mode, False
    has_export_step = any(step.uses.startswith("export.") for step in spec.steps)
    return ("explicit" if has_export_step else "automatic"), True


def run_spec(
    path: str | Path,
    *,
    session: M3Session | None = None,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
    event_sink: EventSink | None = None,
    cancellation_token: CancellationToken | None = None,
) -> PipelineResult:
    """Load a spec file and run it end-to-end, including automatic export.

    This is the entry point for the ``m3resp run <spec.yaml>`` CLI. It injects
    the spec's ``outputs`` and ``experiment`` sections into the context (so steps
    like ``export.rotarc_result`` can read them) and applies the ``outputs:``
    section after the pipeline finishes.

    ``outputs.timestamped`` is resolved exactly once here into
    ``_resolved_output_dir`` (and the raw stamp into ``_run_timestamp``), both
    seeded into context alongside ``_spec_outputs``/``_spec_experiment``. Every
    export path in the run - the automatic export below, built-in steps like
    ``export.rotarc_result``, and any custom export step that reads
    ``_resolved_output_dir`` - shares that one resolved directory, so a run
    never ends up split across two different timestamp folders.

    When ``outputs.dir`` is set and the resolved mode is not ``"none"``, a
    JSON run manifest is written to ``<output_dir>/run_manifest.json``:
    once with ``status: "running"`` before any step executes, then
    atomically replaced with the terminal state - including on failure, so
    a crashed run leaves an honestly-marked-failed manifest rather than
    nothing (Phase 6.3/6.4).
    """

    from m3resp.workflows.manifest import build_manifest, write_manifest_atomic
    from m3resp.workflows.utils import default_run_timestamp, resolve_output_dir

    parsed = load_spec(path)
    mode, inferred = _resolve_output_mode(parsed)
    if inferred:
        warnings.warn(
            f"Pipeline outputs.mode was not set; inferred {mode!r} from the "
            "presence/absence of an 'export.*' step. This becomes required "
            "once 'schema_version' is set.",
            FutureWarning,
            stacklevel=2,
        )

    run_timestamp = default_run_timestamp()
    resolved_output_dir = (
        resolve_output_dir(
            parsed.outputs.dir,
            timestamped=parsed.outputs.timestamped,
            timestamp=run_timestamp,
        )
        if parsed.outputs.dir is not None and mode != "none"
        else None
    )
    extra: dict[str, Any] = {
        "_spec_outputs": parsed.outputs,
        "_spec_experiment": parsed.experiment,
        RESOLVED_OUTPUT_DIR_KEY: resolved_output_dir,
        "_run_timestamp": run_timestamp if parsed.outputs.timestamped else None,
    }

    run_id = new_run_id()
    manifest_path = (
        resolved_output_dir / "run_manifest.json"
        if resolved_output_dir is not None
        else None
    )
    if manifest_path is not None:
        started_at = utc_now_iso()
        write_manifest_atomic(
            manifest_path,
            build_manifest(
                run_id=run_id,
                status="running",
                pipeline_name=parsed.name,
                spec=parsed,
                started_at=started_at,
                output_dir=resolved_output_dir,
            ),
        )

    try:
        result = run_pipeline(
            parsed,
            session=session,
            eit_adapter=eit_adapter,
            emg_adapter=emg_adapter,
            extra_context=extra,
            event_sink=event_sink,
            cancellation_token=cancellation_token,
            run_id=run_id,
        )
    except PipelineExecutionError as exc:
        if manifest_path is not None:
            _write_failed_manifest(manifest_path, parsed, exc)
        raise

    if manifest_path is not None:
        result.manifest_path = _write_result_manifest(
            manifest_path, parsed, result, resolved_output_dir
        )

    _apply_outputs(parsed, result, mode=mode)
    return result


def _write_failed_manifest(
    manifest_path: Path, spec: PipelineSpec, exc: PipelineExecutionError
) -> None:
    """Phase 6.4: a failed run still gets a manifest, honestly marked
    ``"failed"`` - never left as ``"running"`` and never mistaken for a
    success, using whatever step records were gathered before the failure."""

    from m3resp.workflows.manifest import build_manifest, write_manifest_atomic

    write_manifest_atomic(
        manifest_path,
        build_manifest(
            run_id=exc.run_id or "unknown",
            status="failed",
            pipeline_name=spec.name,
            spec=spec,
            started_at=exc.started_at,
            finished_at=utc_now_iso(),
            step_records=exc.step_records,
            error={
                "step_id": exc.step_id,
                "position": exc.position,
                "operation_id": exc.operation_id,
                "message": str(exc),
            },
        ),
    )


def _write_result_manifest(
    manifest_path: Path,
    spec: PipelineSpec,
    result: PipelineResult,
    output_dir: Path | None,
) -> Path:
    """Phase 6.3/6.4: the terminal manifest for a run that returned normally
    - ``"succeeded"`` or ``"cancelled"``, both honestly distinguished from
    ``"failed"`` (see ``_write_failed_manifest``) and from ``"running"``."""

    from m3resp.workflows.manifest import (
        build_manifest,
        collect_input_checksums,
        write_manifest_atomic,
    )

    checksums = collect_input_checksums(result) if spec.outputs.checksums else None
    return write_manifest_atomic(
        manifest_path,
        build_manifest(
            run_id=result.run_id or "unknown",
            status=result.status,
            pipeline_name=spec.name,
            spec=spec,
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_seconds=result.duration_seconds,
            output_dir=output_dir,
            step_records=result.step_records,
            diagnostics=result.diagnostics,
            warnings=result.warnings,
            execution_context=result.execution_context,
            checksums=checksums,
        ),
    )


def _apply_outputs(spec: PipelineSpec, result: PipelineResult, *, mode: str) -> None:
    """Apply the spec's ``outputs:`` section after the pipeline has run.

    ``mode`` (Phase 6.2) replaces the old "any explicit export step present"
    heuristic: ``"none"`` writes nothing, ``"explicit"`` leaves output
    entirely to the spec's own declared export steps (already run during
    execution), and ``"automatic"`` performs session export here - but only
    for a run that actually succeeded (Phase 6.4: never write a success
    summary after a failed or cancelled run).
    """

    out = spec.outputs
    if out.dir is None or mode == "none":
        return

    output_dir = result.context.values.get(RESOLVED_OUTPUT_DIR_KEY) or Path(out.dir)

    if out.figures and result.status == "succeeded":
        output_dir.mkdir(parents=True, exist_ok=True)
        from m3resp.visualization.eit_figures import save_eit_figures

        save_eit_figures(result.context.values, output_dir)

    if mode == "explicit":
        return

    if result.status != "succeeded":
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    session = result.session

    if "eit_sequence" in result.context.values:
        _maybe_assemble_eit(result)

    from m3resp.export.session_export import export_session_summary

    export_session_summary(
        session,
        output_dir,
        summary_json=out.summary_json,
        event_csvs=out.event_csvs,
        parameters_csv=out.parameters_csv,
        postprocessing=out.postprocessing,
        structured_export=out.structured_export,
        processing_run_id=result.processing_run_id,
    )

    _maybe_log_summary(session, output_dir, spec)


def _maybe_assemble_eit(result: PipelineResult) -> None:
    """Populate ``session.processed['eit']`` from the pipeline context."""

    ctx = result.context.values
    if "raw_eit" not in ctx:
        return

    filtered_eit = ctx.get("filtered_eit")
    result.session.processed["eit"] = {
        "sequence": ctx.get("eit_sequence"),
        "raw_eit": ctx.get("raw_eit"),
        "raw_global_impedance": ctx.get("raw_global_impedance"),
        "filter_mode": _infer_filter_mode(filtered_eit),
        "filter_captures": ctx.get("filter_captures", {}),
        "rate_detector": ctx.get("rate_detector"),
        "rate_captures": ctx.get("rate_captures", {}),
        "respiratory_rate_hz": ctx.get("respiratory_rate_hz"),
        "heart_rate_hz": ctx.get("heart_rate_hz"),
        "filtered_eit": filtered_eit,
        "filtered_global_impedance": ctx.get(
            "global_impedance", ctx.get("raw_global_impedance")
        ),
        "breath_intervals": ctx.get("breath_intervals"),
        "continuous_tiv": ctx.get("continuous_tiv"),
        "eeli": ctx.get("eeli"),
        "pixel_tiv": ctx.get("pixel_tiv"),
        "pixel_breaths": ctx.get("pixel_breaths"),
        "tiv_lungspace_mask": ctx.get("tiv_lungspace_mask"),
        "amplitude_lungspace_mask": ctx.get("amplitude_lungspace_mask"),
        "watershed_lungspace_mask": ctx.get("watershed_lungspace_mask"),
        "size_filtered_roi_mask": ctx.get("size_filtered_roi_mask"),
    }


def _infer_filter_mode(filtered_eit: Any) -> str:
    """Derive the filter mode from a filtered EIT signal's label."""

    if filtered_eit is None:
        return "none"
    label = getattr(filtered_eit, "label", "")
    for mode in ("mdn", "lowpass", "bandpass"):
        if mode in label:
            return mode
    return "none"


def _maybe_log_summary(
    session: M3Session, output_dir: Path, spec: PipelineSpec
) -> None:
    """Log a compact workflow summary if loguru is available."""

    try:
        from loguru import logger
        from m3resp.workflows.utils import log_workflow_summary
    except ImportError:
        return

    from m3resp.workflows.summaries import (
        summarize_eit,
        summarize_emg,
        summarize_multimodal,
    )

    has_eit = "eit" in session.processed
    has_emg = "emg" in session.processed
    if has_eit and has_emg:
        summary = summarize_multimodal(session, include_eit=True, include_emg=True)
    elif has_eit:
        summary = summarize_eit(session)
    elif has_emg:
        summary = summarize_emg(session)
    else:
        summary = {}

    if summary:
        log_workflow_summary(spec.name, output_dir, summary)
    else:
        logger.success("{} complete. Output: {}", spec.name, output_dir)


def validate_spec(spec: PipelineSpec, *, available: set[str] | None = None) -> None:
    """Statically check that every step's inputs are produced before use,
    that no two steps write to the same context key without explicit renaming,
    and that every ``@name`` input reference names a declared pipeline input.

    Compatibility wrapper (Phase 3.6) around :func:`collect_diagnostics`:
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
    """Return every independent structural problem in ``spec`` (Phase 3.2/3.3/3.6).

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
            _check_bindings(step_spec, definition, position, step_label, produced)
        )
        diagnostics.extend(
            _check_static_parameters(
                step_spec, definition, spec.inputs, position, step_label
            )
        )

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

    return diagnostics


def _check_bindings(
    step_spec: StepSpec,
    definition: StepDefinition,
    position: int,
    step_label: str,
    produced: dict[str, str],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

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


def _check_static_parameters(
    step_spec: StepSpec,
    definition: StepDefinition,
    spec_inputs: dict[str, Any],
    position: int,
    step_label: str,
) -> list[Diagnostic]:
    """Phase 3.3: validate static parameter values against declared metadata.

    Only checked when the step declares metadata for that parameter name -
    full function-signature coverage is Phase 8.4, not this phase. ``@name``
    references are resolved against ``spec_inputs`` first (an unknown
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


def _replay_captured_warnings(
    caught: list[warnings.WarningMessage],
    record: StepExecutionRecord,
    emit: Any,
) -> None:
    """Store each captured warning on the step's record, emit a
    ``step_warning`` event, and re-raise it through the normal warnings
    machinery (Phase 4.3) so a caller's own filters/``-W error``/
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
    """Phase 5.1: log every executed step onto the session's universal
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
    compiled_step: "CompiledStep", ctx: PipelineContext
) -> dict[str, Any]:
    """Build a step's call kwargs from an already-compiled step (Phase 3.1):
    context reads resolve against the live context; static parameters were
    already fully resolved (``@ref``s and paths) at compile time."""

    kwargs: dict[str, Any] = {
        param: ctx.get(context_key)
        for param, context_key in compiled_step.input_bindings.items()
    }
    kwargs.update(compiled_step.parameters)
    return kwargs


def _store_compiled_outputs(
    compiled_step: "CompiledStep", ctx: PipelineContext, result: Any
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
