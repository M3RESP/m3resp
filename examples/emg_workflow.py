"""Run the EMG-only M3Resp workflow from a Python script."""

from __future__ import annotations

from typing import Any

from toolbox import (
    configure_example_paths,
    load_config,
    save_figures as save_example_figures,
)


REPO_ROOT = configure_example_paths()
cfg = load_config(REPO_ROOT)

from m3resp import M3Session  # noqa: E402
from m3resp.visualization.session import plot_session_overview  # noqa: E402


def summarize_emg(session: M3Session) -> dict[str, Any]:
    """Return a compact summary of the EMG outputs."""

    emg = session.processed["emg"]
    postprocessing = session.parameters["emg_postprocessing"]
    computed = postprocessing["computed"]
    skipped = postprocessing["skipped"]
    ventilator_breaths = computed["event_detection"].get(
        "detect_ventilator_breath",
        [],
    )
    return {
        "channel": emg["channel"],
        "fs": emg["fs"],
        "filter": emg["filter"],
        "n_raw_samples": len(emg["raw_channel"]),
        "n_filtered_samples": len(emg["filtered"]),
        "n_envelope_samples": len(emg["envelope"]),
        "n_emg_breaths": len(session.events["emg_breaths"]),
        "n_ventilator_breaths": len(ventilator_breaths),
        "emg_breath_peak_times": [
            event.peak_time for event in session.events["emg_breaths"]
        ],
        "postprocessing_available": {
            category: len(functions)
            for category, functions in postprocessing["available"].items()
        },
        "postprocessing_computed": {
            category: list(results) for category, results in computed.items() if results
        },
        "postprocessing_skipped": skipped,
    }


def run_workflow() -> tuple[M3Session, dict[str, Any]]:
    """Run the complete EMG workflow and return the session and EMG summary."""

    if not cfg.modules.emg:
        raise RuntimeError(
            "EMG module is disabled in config.yaml – set modules.emg: true to run this workflow."
        )

    session = M3Session()
    session.load_emg(cfg.emg.file, verbose=False)
    session.preprocess_emg()
    session.detect_emg_breaths()

    ventilator = None
    if cfg.modules.vent and cfg.vent.file is not None:
        ventilator = session.emg_adapter.load(str(cfg.vent.file), verbose=False)
    session.postprocess_emg(ventilator=ventilator)

    session.export_summary(cfg.output.emg_only)

    return session, summarize_emg(session)


def save_figures(session: M3Session) -> None:
    """Save the EMG overview figure."""

    fig = plot_session_overview(session, max_seconds=120)
    save_example_figures(cfg.output.emg_only, {"overview.png": fig})


def main() -> None:
    """Run the example and print a compact EMG summary."""

    session, emg_summary = run_workflow()
    save_figures(session)

    print("EMG workflow complete.")
    print(f"Output directory: {cfg.output.emg_only}")
    for key, value in emg_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
