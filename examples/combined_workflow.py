"""Run the multimodal M3Resp workflow from a Python script."""

from __future__ import annotations

from typing import Any

from toolbox import (
    configure_example_paths,
    load_config,
    save_figures as save_example_figures,
)


REPO_ROOT = configure_example_paths("eitprocessing")
cfg = load_config(REPO_ROOT)

from m3resp import M3Session  # noqa: E402
from m3resp.visualization.session import (  # noqa: E402
    plot_eit_processing_summary,
    plot_session_overview,
)


def run_workflow() -> tuple[M3Session, dict[str, Any]]:
    """Run the complete multimodal workflow and return the session and summary."""

    session = M3Session()

    if cfg.modules.eit:
        session.load_eit(cfg.eit.file, vendor=cfg.eit.vendor)
        session.preprocess_eit()
        session.detect_eit_breaths()

    if cfg.modules.emg:
        session.load_emg(cfg.emg.file, verbose=False)
        session.preprocess_emg()
        session.detect_emg_breaths()
        ventilator = None
        if cfg.modules.vent and cfg.vent.file is not None:
            ventilator = session.emg_adapter.load(str(cfg.vent.file), verbose=False)
        session.postprocess_emg(ventilator=ventilator)

    if cfg.modules.eit and cfg.modules.emg:
        session.align_modalities(
            method=cfg.alignment.method,
            offset_seconds=cfg.alignment.manual_offset_seconds,
        )

    session.export_summary(cfg.output.combined)

    # ------------------------------------------------------------------
    # Build summary dict (gracefully handle disabled modules)
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {}

    if cfg.modules.eit:
        eit = session.processed["eit"]
        summary.update(
            {
                "filter_mode": eit["filter_mode"],
                "respiratory_rate_bpm": float(eit["respiratory_rate_hz"] * 60),
                "heart_rate_bpm": float(eit["heart_rate_hz"] * 60),
                "n_eit_breaths": len(session.events["eit_breaths"]),
                "n_continuous_tiv_values": len(eit["continuous_tiv"]),
                "n_eeli_values": len(eit["eeli"]),
                "pixel_tiv_shape_per_breath": None
                if eit["pixel_tiv"] is None or len(eit["pixel_tiv"].values) == 0
                else eit["pixel_tiv"].values[0].shape,
            }
        )

    if cfg.modules.emg:
        postprocessing = session.parameters["emg_postprocessing"]
        ventilator_breaths = postprocessing["computed"]["event_detection"].get(
            "detect_ventilator_breath", []
        )
        summary.update(
            {
                "n_emg_breaths": len(session.events["emg_breaths"]),
                "n_ventilator_breaths": len(ventilator_breaths),
                "emg_postprocessing_available": {
                    category: len(functions)
                    for category, functions in postprocessing["available"].items()
                },
                "emg_postprocessing_computed": {
                    category: list(results)
                    for category, results in postprocessing["computed"].items()
                    if results
                },
                "emg_postprocessing_skipped": postprocessing["skipped"],
            }
        )

    return session, summary


def save_figures(session: M3Session) -> None:
    """Save overview and EIT processing figures (respects module switches)."""

    figures: dict[str, Any] = {}

    if cfg.modules.emg:
        figures["overview.png"] = plot_session_overview(session, max_seconds=120)

    if cfg.modules.eit:
        figures["eit-processing.png"] = plot_eit_processing_summary(session)
        figures["eit-rate-detection.png"] = session.processed["eit"][
            "rate_detector"
        ].plotting.plot(**session.processed["eit"]["rate_captures"])

    save_example_figures(cfg.output.combined, figures)


def main() -> None:
    """Run the example and print a compact summary."""

    session, summary = run_workflow()
    save_figures(session)

    print("Multimodal workflow complete.")
    print(f"Output directory: {cfg.output.combined}")
    print(
        f"Active modules: eit={cfg.modules.eit}, emg={cfg.modules.emg}, vent={cfg.modules.vent}"
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
