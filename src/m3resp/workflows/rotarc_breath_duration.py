"""ROTARC-style EIT breath-duration variability workflow.

The processing pipeline is assembled declaratively from registered steps via the
:mod:`m3resp.pipeline` engine instead of a hand-wired preprocess function. The
equivalent spec ships at ``examples/ROTARC_example/breath-duration.pipeline.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from m3resp import M3Session
from m3resp.core.config import WorkflowConfig
from m3resp.export.session_export import export_session_summary
from m3resp.pipeline import run_pipeline
from m3resp.workflows.configured import WorkflowResult, coerce_workflow_config
from m3resp.workflows.toolbox import subject_result_filename, write_json


def run_rotarc_breath_duration_workflow(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
    eit_adapter: Any = None,
) -> WorkflowResult:
    """Run the ROTARC breath-duration CV calculation.

    This function uses ``M3Session`` for loading, event normalization, storage, and
    selected exports.
    """

    cfg = coerce_workflow_config(config, root=root)
    cfg.validate_rotarc()

    session, summary = _run_rotarc_eit_pipeline(
        cfg,
        eit_adapter=eit_adapter,
    )

    output_dir = Path(
        os.path.join(
            str(cfg.output.combined),
            "subject_results",
            str(cfg.rotarc.run_identifier),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(
        os.path.join(
            output_dir,
            subject_result_filename(
                str(cfg.rotarc.subject_id),
                str(cfg.rotarc.mode),
                cfg.rotarc.timepoint,
                str(cfg.rotarc.selection),
            ),
        )
    )
    result_path.write_text(f"{summary['breath_duration_cv']:.8f}", encoding="utf-8")
    summary["result_path"] = str(result_path)

    export_session_summary(
        session,
        output_dir,
        summary_json=cfg.results.summary_json,
        event_csvs=cfg.results.event_csvs,
        parameters_csv=cfg.results.parameters_csv,
        postprocessing=cfg.results.postprocessing,
    )
    write_json(Path(os.path.join(output_dir, "rotarc_summary.json")), summary)

    return WorkflowResult(
        session=session,
        summary=summary,
        output_dir=output_dir,
        figures={},
    )


def build_rotarc_spec(cfg: WorkflowConfig) -> dict[str, Any]:
    """Build the declarative breath-duration pipeline spec from a workflow config.

    The ``selection`` setting decides whether breath detection runs on the sliced
    window (``selected``) or the full filtered global impedance signal.
    """

    rotarc = cfg.rotarc
    processing = cfg.eit.processing
    slice_with = {
        "start": rotarc.start,
        "end": rotarc.end,
        "mode": rotarc.slicing_mode,
    }

    steps: list[dict[str, Any]] = [
        {
            "uses": "eit.load",
            "with": {"file": str(cfg.eit.file), "vendor": cfg.eit.vendor},
        },
        {
            "uses": "eit.slice",
            "in": {"signal": "raw_eit"},
            "with": slice_with,
            "out": {"result": "selected_eit"},
        },
        {
            "uses": "eit.detect_rates",
            "in": {"signal": "selected_eit"},
            "with": {"subject_type": processing.subject_type},
        },
        {"uses": "eit.mdn_filter", "in": {"signal": "raw_eit"}},
        {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
    ]

    detect = {
        "uses": "eit.detect_breaths",
        "with": {"min_duration_s": processing.breath_min_duration_seconds},
    }
    if rotarc.selection == "selected":
        steps.append(
            {
                "uses": "eit.slice",
                "in": {"signal": "global_impedance"},
                "with": slice_with,
                "out": {"result": "detection_signal"},
            }
        )
        detect["in"] = {"signal": "detection_signal"}
    else:
        detect["in"] = {"signal": "global_impedance"}
    steps.append(detect)
    steps.append({"uses": "eit.normalize_breaths"})

    steps.append(
        {"uses": "metric.interval_cv", "in": {"intervals": "breath_intervals"}}
    )

    return {"name": "rotarc-breath-duration", "steps": steps}


def _run_rotarc_eit_pipeline(
    cfg: WorkflowConfig,
    *,
    eit_adapter: Any,
) -> tuple[M3Session, dict[str, Any]]:
    session = M3Session(eit_adapter=eit_adapter)
    result = run_pipeline(build_rotarc_spec(cfg), session=session)
    context = result.context

    session.parameters["rotarc_breath_duration"] = {
        "data_path": str(cfg.eit.file),
        "subject_id": cfg.rotarc.subject_id,
        "mode": cfg.rotarc.mode,
        "timepoint": cfg.rotarc.timepoint,
        "selection": cfg.rotarc.selection,
        "slicing_mode": cfg.rotarc.slicing_mode,
        "start": cfg.rotarc.start,
        "end": cfg.rotarc.end,
        "subject_type": cfg.eit.processing.subject_type,
        "breath_min_duration_seconds": cfg.eit.processing.breath_min_duration_seconds,
    }

    summary = {
        **session.parameters["rotarc_breath_duration"],
        "respiratory_rate_hz": float(context.get("respiratory_rate_hz")),
        "heart_rate_hz": float(context.get("heart_rate_hz")),
        "n_breaths": int(context.get("n")),
        "mean_breath_duration_seconds": float(context.get("mean")),
        "std_breath_duration_seconds": float(context.get("std")),
        "breath_duration_cv": float(context.get("cv")),
    }
    return session, summary
