"""Execution engine for declarative M3Resp pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session
from m3resp.workflows.context import SESSION_KEY, PipelineContext
from m3resp.workflows.registry import StepDefinition, get_step
from m3resp.workflows.spec import PipelineSpec, StepSpec, load_spec

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
    pipeline_result = PipelineResult(name=parsed.name, context=ctx, outputs=produced)
    if ctx.session.datamodel is not None:
        run = ctx.session.datamodel.record_pipeline_result(pipeline_result)
        pipeline_result.processing_run_id = run.processing_run_id
    return pipeline_result


def run_spec(
    path: str | Path,
    *,
    session: M3Session | None = None,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
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
    """

    from m3resp.workflows.utils import default_run_timestamp, resolve_output_dir

    parsed = load_spec(path)
    run_timestamp = default_run_timestamp()
    resolved_output_dir = (
        resolve_output_dir(
            parsed.outputs.dir,
            timestamped=parsed.outputs.timestamped,
            timestamp=run_timestamp,
        )
        if parsed.outputs.dir is not None
        else None
    )
    extra: dict[str, Any] = {
        "_spec_outputs": parsed.outputs,
        "_spec_experiment": parsed.experiment,
        "_resolved_output_dir": resolved_output_dir,
        "_run_timestamp": run_timestamp if parsed.outputs.timestamped else None,
    }
    result = run_pipeline(
        parsed,
        session=session,
        eit_adapter=eit_adapter,
        emg_adapter=emg_adapter,
        extra_context=extra,
    )
    _apply_outputs(parsed, result)
    return result


def _apply_outputs(spec: PipelineSpec, result: PipelineResult) -> None:
    """Apply the spec's ``outputs:`` section after the pipeline has run.

    If ``outputs.dir`` is set and the spec does not already contain an
    explicit ``export.session_summary`` or ``export.rotarc_result`` step,
    this function performs the export automatically.
    """

    out = spec.outputs
    if out.dir is None:
        return

    explicit_export_steps = {
        "export.session_summary",
        "export.rotarc_result",
        "export.json_file",
    }
    step_names = {s.uses for s in spec.steps}
    if step_names & explicit_export_steps:
        return

    output_dir = result.context.values.get("_resolved_output_dir") or Path(out.dir)
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
    and that no two steps write to the same context key without explicit renaming.

    ``available`` lists extra context keys seeded outside the spec.
    """

    _ensure_steps_registered()
    # _spec_outputs, _spec_experiment, _resolved_output_dir, and _run_timestamp
    # are always injected by run_spec before the pipeline executes, so treat
    # them as globally available. Pre-seeded keys are exempt from
    # duplicate-write detection.
    seeded: set[str] = {
        SESSION_KEY,
        "_spec_outputs",
        "_spec_experiment",
        "_resolved_output_dir",
        "_run_timestamp",
        *spec.inputs,
        *(available or set()),
    }
    # Maps context key -> label of the step that produced it, for error messages.
    produced: dict[str, str] = {key: "pipeline seed" for key in seeded}

    for position, step_spec in enumerate(spec.steps):
        definition = get_step(step_spec.uses)
        for context_key in _required_context_keys(definition, step_spec):
            if context_key not in produced:
                raise PipelineSpecError(
                    f"Step #{position} '{step_spec.uses}' reads context key "
                    f"'{context_key}', which is not produced by an earlier step "
                    f"or declared in inputs."
                )
        step_label = f"step #{position} '{step_spec.uses}'"
        for context_key in _output_context_keys(definition, step_spec):
            if context_key in produced and context_key not in seeded:
                raise PipelineSpecError(
                    f"Step #{position} '{step_spec.uses}' writes to context key "
                    f"'{context_key}', which was already produced by "
                    f"{produced[context_key]}. Use 'out:' to rename one of the outputs."
                )
            produced[context_key] = step_label


def _required_context_keys(definition: StepDefinition, step_spec: StepSpec) -> set[str]:
    keys = set()
    for param, default in definition.reads.items():
        context_key = step_spec.inputs.get(param, default)
        if context_key is None:
            raise PipelineSpecError(
                f"Step '{step_spec.uses}' requires an explicit 'in: {{{param}: ...}}' "
                f"binding — this parameter has no default context key."
            )
        keys.add(context_key)
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
        if context_key is None:
            raise PipelineSpecError(
                f"Step #{position} '{step_spec.uses}' requires an explicit "
                f"'in: {{{param}: ...}}' binding — this parameter has no default context key."
            )
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
