"""ROTARC-style EIT breath-duration variability workflow."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from m3resp import M3Session
from m3resp.core.config import WorkflowConfig
from m3resp.export.session_export import export_session_summary
from m3resp.workflows.configured import WorkflowResult
from m3resp.workflows.toolbox import (
    coerce_slice_value,
    load_workflow_config_with_raw,
    slice_by_index,
    slice_by_time,
    subject_result_filename,
    write_json,
)


def run_rotarc_breath_duration_workflow(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
    data_path: str | Path | None = None,
    subject_id: str | None = None,
    mode: str | None = None,
    start: int | float | None = None,
    end: int | float | None = None,
    timepoint: str | None = None,
    slicing_mode: str | None = None,
    selection: str | None = None,
    run_identifier: str | None = None,
    export: bool = True,
    eit_adapter: Any = None,
) -> WorkflowResult:
    """Run the ROTARC breath-duration CV calculation.

    This function uses``M3Session`` for loading, event normalization, storage, and
    selected exports.
    """

    cfg, raw_config = load_workflow_config_with_raw(config, root=root)
    rotarc = dict(raw_config.get("rotarc", {}))
    run_config = _resolve_run_config(
        cfg,
        rotarc,
        data_path=data_path,
        subject_id=subject_id,
        mode=mode,
        start=start,
        end=end,
        timepoint=timepoint,
        slicing_mode=slicing_mode,
        selection=selection,
        run_identifier=run_identifier,
    )

    session, summary = _run_rotarc_eit_pipeline(
        cfg,
        run_config,
        eit_adapter=eit_adapter,
    )

    output_dir = Path(
        os.path.join(
            str(cfg.output.combined),
            "subject_results",
            str(run_config["run_identifier"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(
        os.path.join(
            output_dir,
            subject_result_filename(
                str(run_config["subject_id"]),
                str(run_config["mode"]),
                run_config.get("timepoint"),
                str(run_config["selection"]),
            ),
        )
    )
    result_path.write_text(f"{summary['breath_duration_cv']:.8f}", encoding="utf-8")
    summary["result_path"] = str(result_path)

    if export:
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
    run_config: Mapping[str, Any],
    *,
    eit_adapter: Any,
) -> tuple[M3Session, dict[str, Any]]:
    session = M3Session(eit_adapter=eit_adapter)
    session.load_eit(
        run_config["data_path"],
        vendor=cfg.eit.vendor,
    )
    processed = session.preprocess_eit(
        preprocess=_preprocess_rotarc_eit,
        subject_type=cfg.eit.processing.subject_type,
        breath_min_duration_seconds=cfg.eit.processing.breath_min_duration_seconds,
        start=run_config["start"],
        end=run_config["end"],
        slicing_mode=str(run_config["slicing_mode"]),
        selection=str(run_config["selection"]),
    )
    session.detect_eit_breaths()
    session.parameters["rotarc_breath_duration"] = {
        "data_path": str(run_config["data_path"]),
        "subject_id": run_config["subject_id"],
        "mode": run_config["mode"],
        "timepoint": run_config.get("timepoint"),
        "selection": str(run_config["selection"]),
        "slicing_mode": str(run_config["slicing_mode"]),
        "start": run_config["start"],
        "end": run_config["end"],
        "subject_type": cfg.eit.processing.subject_type,
        "breath_min_duration_seconds": cfg.eit.processing.breath_min_duration_seconds,
    }

    breath_intervals = processed["breath_intervals"]
    breath_durations = np.asarray(
        [interval[1] - interval[0] for interval in breath_intervals.intervals],
        dtype=float,
    )
    breath_duration_cv = float(breath_durations.std() / breath_durations.mean())
    if math.isnan(breath_duration_cv):
        raise ValueError("Breath duration CV is NaN; no valid breaths were detected.")

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
    selected_eit = _slice_rotarc_signal(
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
        detection_signal = _slice_rotarc_signal(
            filtered_global_impedance,
            start=start,
            end=end,
            slicing_mode=slicing_mode,
        )
    elif selection in {"all", "entire", "full"}:
        detection_signal = filtered_global_impedance
    else:
        raise ValueError(
            "rotarc.selection must be one of: selected, all, entire, full."
        )

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


def _slice_rotarc_signal(
    data: Any,
    *,
    start: int | float,
    end: int | float,
    slicing_mode: str,
) -> Any:
    if slicing_mode == "index":
        return slice_by_index(data, start=int(start), end=int(end))
    if slicing_mode == "time":
        return slice_by_time(data, start=float(start), end=float(end))
    raise ValueError("slicing_mode must be 'index' or 'time'.")


def _resolve_run_config(
    cfg: WorkflowConfig,
    rotarc: Mapping[str, Any],
    *,
    data_path: str | Path | None,
    subject_id: str | None,
    mode: str | None,
    start: int | float | None,
    end: int | float | None,
    timepoint: str | None,
    slicing_mode: str | None,
    selection: str | None,
    run_identifier: str | None,
) -> dict[str, Any]:
    resolved_data_path = data_path if data_path is not None else cfg.eit.file
    if resolved_data_path is None:
        raise ValueError("ROTARC workflow requires eit.file or data_path.")

    resolved_slicing_mode = slicing_mode or str(rotarc.get("slicing_mode", "index"))
    return {
        "data_path": str(resolved_data_path),
        "subject_id": subject_id or str(rotarc.get("subject_id", "subject")),
        "mode": mode or str(rotarc.get("mode", "mode")),
        "timepoint": timepoint if timepoint is not None else rotarc.get("timepoint"),
        "start": coerce_slice_value(
            start,
            rotarc.get("start"),
            resolved_slicing_mode,
            value_name="start slice value",
        ),
        "end": coerce_slice_value(
            end,
            rotarc.get("end"),
            resolved_slicing_mode,
            value_name="end slice value",
        ),
        "slicing_mode": resolved_slicing_mode,
        "selection": selection or str(rotarc.get("selection", "selected")),
        "run_identifier": run_identifier or str(rotarc.get("run_identifier", "rotarc")),
    }
