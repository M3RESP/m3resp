"""Run the EMG-only M3Resp workflow from a Python script."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    """Find the local m3resp repository root."""

    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "m3resp"
        ).exists():
            return candidate
    raise RuntimeError("Could not find the m3resp repository root.")


REPO_ROOT = find_repo_root()

src_path = REPO_ROOT / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from m3resp import M3Session  # noqa: E402
from m3resp.adapters import ReSurfEMGAdapter  # noqa: E402
from m3resp.visualization.session import plot_session_overview  # noqa: E402


EMG_FILE = REPO_ROOT / "data" / "source" / "emg_data_synth_quiet_breathing.Poly5"

EMG_CHANNEL = 0
EMG_HIGH_PASS_HZ = 80
EMG_LOW_PASS_HZ = None
EMG_ENVELOPE_WINDOW_SECONDS = 0.5
EMG_MIN_BREATH_WIDTH_SECONDS = 1.0

OUTPUT_DIR = REPO_ROOT / "output" / "emg-summary"


def load_emg(path: str, **kwargs: Any) -> dict[str, Any]:
    """Load an EMG file with ReSurfEMG's generic converter."""

    from resurfemg.data_connector.converter_functions import load_file

    array, dataframe, metadata = load_file(path, **kwargs)
    return {
        "array": array,
        "dataframe": dataframe,
        "metadata": metadata,
    }


def preprocess_emg(
    recording: dict[str, Any],
    channel: int = EMG_CHANNEL,
    high_pass_hz: float = EMG_HIGH_PASS_HZ,
    low_pass_hz: float | None = EMG_LOW_PASS_HZ,
    envelope_window_seconds: float = EMG_ENVELOPE_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Band-pass one EMG channel and compute an ARV envelope."""

    import numpy as np
    from resurfemg.preprocessing.envelope import full_rolling_arv
    from resurfemg.preprocessing.filtering import emg_bandpass_butter

    if not isinstance(recording, dict) or "array" not in recording:
        raise TypeError("preprocess_emg expects a ReSurfEMG EMG recording dict.")
    if "metadata" not in recording:
        raise TypeError("preprocess_emg expects EMG recording metadata.")

    metadata = dict(recording["metadata"])
    fs = float(metadata["fs"])
    array = recording["array"]
    raw = np.asarray(array[channel], dtype=float)

    if low_pass_hz is None:
        low_pass_hz = min(fs / 2 * 0.95, 500)

    filtered = emg_bandpass_butter(
        emg_raw=raw,
        high_pass=high_pass_hz,
        low_pass=low_pass_hz,
        fs_emg=fs,
    )
    envelope_window_samples = max(1, int(envelope_window_seconds * fs))
    envelope = full_rolling_arv(filtered, envelope_window_samples)

    return {
        **recording,
        "channel": channel,
        "fs": fs,
        "raw_channel": raw,
        "filtered": filtered,
        "envelope": envelope,
        "filter": {
            "high_pass_hz": high_pass_hz,
            "low_pass_hz": low_pass_hz,
            "envelope_window_seconds": envelope_window_seconds,
        },
    }


def detect_emg_breaths(
    processed_emg: dict[str, Any],
    min_breath_width_seconds: float = EMG_MIN_BREATH_WIDTH_SECONDS,
    half_window_seconds: float = 0.5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Detect EMG breath peaks and convert them to M3Resp breath rows."""

    from resurfemg.postprocessing.event_detection import detect_emg_breaths

    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]
    min_width_samples = max(1, int(min_breath_width_seconds * fs))
    half_window_samples = max(1, int(half_window_seconds * fs))

    peak_indices = detect_emg_breaths(
        emg_env=envelope,
        min_peak_width_s=min_width_samples,
        **kwargs,
    )

    events = []
    for peak_index in peak_indices:
        start_index = max(0, int(peak_index) - half_window_samples)
        end_index = min(len(envelope) - 1, int(peak_index) + half_window_samples)
        events.append(
            {
                "start_time": start_index / fs,
                "end_time": end_index / fs,
                "peak_time": int(peak_index) / fs,
                "source": "resurfemg.detect_emg_breaths",
                "metadata": {
                    "start_index": start_index,
                    "peak_index": int(peak_index),
                    "end_index": end_index,
                    "channel": processed_emg["channel"],
                },
            }
        )

    return events


def summarize_emg(session: M3Session) -> dict[str, Any]:
    """Return a compact summary of the EMG outputs."""

    emg = session.processed["emg"]
    return {
        "channel": emg["channel"],
        "fs": emg["fs"],
        "filter": emg["filter"],
        "n_raw_samples": len(emg["raw_channel"]),
        "n_filtered_samples": len(emg["filtered"]),
        "n_envelope_samples": len(emg["envelope"]),
        "n_emg_breaths": len(session.events["emg_breaths"]),
        "emg_breath_peak_times": [
            event.peak_time for event in session.events["emg_breaths"]
        ],
    }


def run_workflow() -> tuple[M3Session, dict[str, Any]]:
    """Run the complete EMG workflow and return the session and EMG summary."""

    session = M3Session(emg_adapter=ReSurfEMGAdapter(loader=load_emg))

    session.load_emg(EMG_FILE, verbose=False)
    session.preprocess_emg(preprocess=preprocess_emg)
    session.detect_emg_breaths(detector=detect_emg_breaths)
    session.export_summary(OUTPUT_DIR)

    return session, summarize_emg(session)


def save_figures(session: M3Session) -> None:
    """Save the EMG overview figure."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plot_session_overview(session, max_seconds=120)
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=150)


def main() -> None:
    """Run the example and print a compact EMG summary."""

    session, emg_summary = run_workflow()
    save_figures(session)

    print("EMG workflow complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    for key, value in emg_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
