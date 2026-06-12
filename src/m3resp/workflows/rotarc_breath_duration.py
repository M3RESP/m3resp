"""ROTARC-style EIT breath-duration variability workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from m3resp import M3Session
from m3resp.core.config import WorkflowConfig
from m3resp.export.session_export import export_session_summary
from m3resp.workflows.configured import WorkflowResult, coerce_workflow_config
from m3resp.workflows.toolbox import (
    slice_signal_by_mode,
    subject_result_filename,
    write_json,
)


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


def _run_rotarc_eit_pipeline(
    cfg: WorkflowConfig,
    *,
    eit_adapter: Any,
) -> tuple[M3Session, dict[str, Any]]:
    session = M3Session(eit_adapter=eit_adapter)
    session.load_eit(
        cfg.eit.file,
        vendor=cfg.eit.vendor,
    )
    processed = session.preprocess_eit(
        preprocess=_preprocess_rotarc_eit,
        subject_type=cfg.eit.processing.subject_type,
        breath_min_duration_seconds=cfg.eit.processing.breath_min_duration_seconds,
        start=cfg.rotarc.start,
        end=cfg.rotarc.end,
        slicing_mode=cfg.rotarc.slicing_mode,
        selection=cfg.rotarc.selection,
    )
    session.detect_eit_breaths()
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

    breath_intervals = processed["breath_intervals"]
    breath_durations = np.asarray(
        [interval[1] - interval[0] for interval in breath_intervals.intervals],
        dtype=float,
    )
    breath_duration_cv = float(breath_durations.std() / breath_durations.mean())

    summary = {
        **session.parameters["rotarc_breath_duration"],
        "respiratory_rate_hz": float(processed["respiratory_rate_hz"]),
        "heart_rate_hz": float(processed["heart_rate_hz"]),
        "n_breaths": int(len(breath_intervals.intervals)),
        "mean_breath_duration_seconds": float(breath_durations.mean()),
        "std_breath_duration_seconds": float(breath_durations.std()),
        "breath_duration_cv": breath_duration_cv,
    }
    return session, summary


def _preprocess_rotarc_eit(
    sequence: Any,
    *,
    subject_type: str,
    breath_min_duration_seconds: float,
    start: int | float,
    end: int | float,
    slicing_mode: str,
    selection: str,
) -> dict[str, Any]:
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.rate_detection import RateDetection
    from eitprocessing.filters.mdn import MDNFilter

    raw_eit = sequence.data["raw"]
    selected_eit = slice_signal_by_mode(
        raw_eit,
        start=start,
        end=end,
        slicing_mode=slicing_mode,
    )
    respiratory_rate_hz, heart_rate_hz = RateDetection(subject_type).apply(selected_eit)

    filtered_eit = MDNFilter(
        respiratory_rate=respiratory_rate_hz,
        heart_rate=heart_rate_hz,
    ).apply(sequence.data["raw"], label="filtered")
    sequence.eit_data.add(filtered_eit, overwrite=True)

    filtered_global_impedance = filtered_eit.get_summed_impedance()
    sequence.continuous_data.add(filtered_global_impedance, overwrite=True)

    if selection == "selected":
        detection_signal = slice_signal_by_mode(
            filtered_global_impedance,
            start=start,
            end=end,
            slicing_mode=slicing_mode,
        )
    else:
        detection_signal = filtered_global_impedance

    breath_intervals = BreathDetection(
        minimum_duration=breath_min_duration_seconds
    ).find_breaths(detection_signal)
    return {
        "sequence": sequence,
        "filtered_eit": filtered_eit,
        "filtered_global_impedance": filtered_global_impedance,
        "respiratory_rate_hz": respiratory_rate_hz,
        "heart_rate_hz": heart_rate_hz,
        "breath_intervals": breath_intervals,
    }
