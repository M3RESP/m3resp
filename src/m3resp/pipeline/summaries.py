"""Compact session summaries for logging after a pipeline run."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session


def summarize_eit(session: M3Session) -> dict[str, Any]:
    eit = session.processed["eit"]
    summary: dict[str, Any] = {"filter_mode": eit["filter_mode"]}
    if eit.get("respiratory_rate_hz") is not None:
        summary["respiratory_rate_bpm"] = float(eit["respiratory_rate_hz"] * 60)
    if eit.get("heart_rate_hz") is not None:
        summary["heart_rate_bpm"] = float(eit["heart_rate_hz"] * 60)
    if "eit_breaths" in session.events:
        summary["n_eit_breaths"] = len(session.events["eit_breaths"])
    if eit.get("continuous_tiv") is not None:
        summary["n_continuous_tiv_values"] = len(eit["continuous_tiv"])
    if eit.get("eeli") is not None:
        summary["n_eeli_values"] = len(eit["eeli"])
    if "pixel_tiv" in eit:
        summary["pixel_tiv_shape_per_breath"] = (
            None
            if eit["pixel_tiv"] is None or len(eit["pixel_tiv"].values) == 0
            else eit["pixel_tiv"].values[0].shape
        )
    return summary


def summarize_emg(session: M3Session) -> dict[str, Any]:
    emg = session.processed["emg"]
    postprocessing = session.parameters.get("emg_postprocessing", {})
    computed = postprocessing.get("computed", {})
    ventilator_breaths = computed.get("event_detection", {}).get(
        "detect_ventilator_breath", []
    )
    summary = {
        "channel": emg["channel"],
        "fs": emg["fs"],
        "filter": emg["filter"],
        "n_raw_samples": len(emg["raw_channel"]),
        "n_filtered_samples": len(emg["filtered"]),
        "n_envelope_samples": len(emg["envelope"]),
        "n_ventilator_breaths": len(ventilator_breaths),
    }
    if "emg_breaths" in session.events:
        summary["n_emg_breaths"] = len(session.events["emg_breaths"])
    if postprocessing:
        pp = postprocessing
        summary["postprocessing_available"] = {
            category: len(functions)
            for category, functions in pp.get("available", {}).items()
        }
        summary["postprocessing_computed"] = {
            category: list(results)
            for category, results in pp.get("computed", {}).items()
            if results
        }
        summary["postprocessing_skipped"] = pp.get("skipped", {})
    return summary


def summarize_multimodal(
    session: M3Session,
    *,
    include_eit: bool,
    include_emg: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if include_eit:
        summary.update(summarize_eit(session))
    if include_emg:
        emg_summary = summarize_emg(session)
        for key in ("n_emg_breaths", "n_ventilator_breaths"):
            if key in emg_summary:
                summary[key] = emg_summary[key]
        for key in (
            "postprocessing_available",
            "postprocessing_computed",
            "postprocessing_skipped",
        ):
            if key in emg_summary:
                summary[f"emg_{key}"] = emg_summary[key]
    return summary
