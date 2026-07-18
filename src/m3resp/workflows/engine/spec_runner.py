"""Spec-file loading, running, and output/manifest writing (`run_spec`)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.context import (
    RESOLVED_OUTPUT_DIR_KEY,
)
from m3resp.workflows.lifecycle import (
    CancellationToken,
    EventSink,
    PipelineExecutionError,
    new_run_id,
    utc_now_iso,
)
from m3resp.workflows.spec import PipelineSpec, load_spec

from ._shared import PipelineResult
from .execution import run_pipeline


def _resolve_output_mode(spec: PipelineSpec) -> tuple[str, bool]:
    """replaces the old "any of three hardcoded export step names
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
    nothing.
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
    """a failed run still gets a manifest, honestly marked
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
    """the terminal manifest for a run that returned normally
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

    ``mode`` replaces the old "any explicit export step present"
    heuristic: ``"none"`` writes nothing, ``"explicit"`` leaves output
    entirely to the spec's own declared export steps (already run during
    execution), and ``"automatic"`` performs session export here - but only
    for a run that actually succeeded (never write a success
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
