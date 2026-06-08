"""Run the EIT-only M3Resp workflow from a Python script."""

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
from m3resp.visualization.session import plot_eit_processing_summary  # noqa: E402


def summarize_eit(session: M3Session) -> dict[str, Any]:
    """Return the same EIT summary fields as the multimodal workflow."""

    eit = session.processed["eit"]
    return {
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


def run_workflow() -> tuple[M3Session, dict[str, Any]]:
    """Run the complete EIT workflow and return the session and EIT summary."""

    if not cfg.modules.eit:
        raise RuntimeError(
            "EIT module is disabled in config.yaml – set modules.eit: true to run this workflow."
        )

    session = M3Session()
    session.load_eit(cfg.eit.file, vendor=cfg.eit.vendor)
    session.preprocess_eit()
    session.detect_eit_breaths()
    session.export_summary(cfg.output.eit_only)

    return session, summarize_eit(session)


def save_figures(session: M3Session) -> None:
    """Save EIT processing figures."""

    eit_fig = plot_eit_processing_summary(session)
    rate_fig = session.processed["eit"]["rate_detector"].plotting.plot(
        **session.processed["eit"]["rate_captures"]
    )
    save_example_figures(
        cfg.output.eit_only,
        {
            "eit-processing.png": eit_fig,
            "eit-rate-detection.png": rate_fig,
        },
    )


def main() -> None:
    """Run the example and print a compact EIT summary."""

    session, eit_summary = run_workflow()
    save_figures(session)

    print("EIT workflow complete.")
    print(f"Output directory: {cfg.output.eit_only}")
    for key, value in eit_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
