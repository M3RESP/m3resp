"""Summary helpers for YAML-configured workflows."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session


def summarize_eit(session: M3Session) -> dict[str, Any]:
    """Return a compact summary of processed EIT outputs."""

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


def summarize_emg_postprocessing(
    session: M3Session,
    *,
    key_prefix: str = "postprocessing_",
) -> dict[str, Any]:
    """Return compact EMG postprocessing metadata."""

    postprocessing = session.parameters["emg_postprocessing"]
    return {
        f"{key_prefix}available": {
            category: len(functions)
            for category, functions in postprocessing["available"].items()
        },
        f"{key_prefix}computed": {
            category: list(results)
            for category, results in postprocessing["computed"].items()
            if results
        },
        f"{key_prefix}skipped": postprocessing["skipped"],
    }


def summarize_emg(session: M3Session) -> dict[str, Any]:
    """Return a compact summary of processed EMG outputs."""

    emg = session.processed["emg"]
    postprocessing = session.parameters.get("emg_postprocessing", {})
    computed = postprocessing.get("computed", {})
    ventilator_breaths = computed.get("event_detection", {}).get(
        "detect_ventilator_breath",
        [],
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
        summary["emg_breath_peak_times"] = [
            event.peak_time for event in session.events["emg_breaths"]
        ]
    if postprocessing:
        summary.update(summarize_emg_postprocessing(session))
    return summary


def summarize_multimodal(
    session: M3Session,
    *,
    include_eit: bool,
    include_emg: bool,
) -> dict[str, Any]:
    """Return the compact summary used by configured multimodal workflows."""

    summary: dict[str, Any] = {}
    if include_eit:
        summary.update(summarize_eit(session))
    if include_emg:
        emg_summary = summarize_emg(session)
        for key in ("n_emg_breaths", "n_ventilator_breaths"):
            if key in emg_summary:
                summary[key] = emg_summary[key]
        if "postprocessing_available" in emg_summary:
            summary["emg_postprocessing_available"] = emg_summary[
                "postprocessing_available"
            ]
        if "postprocessing_computed" in emg_summary:
            summary["emg_postprocessing_computed"] = emg_summary[
                "postprocessing_computed"
            ]
        if "postprocessing_skipped" in emg_summary:
            summary["emg_postprocessing_skipped"] = emg_summary[
                "postprocessing_skipped"
            ]
    return summary
