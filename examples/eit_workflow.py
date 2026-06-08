"""Run the EIT-only M3Resp workflow from a Python script."""

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
ORG_ROOT = REPO_ROOT.parent

for path in [
    REPO_ROOT / "src",
    ORG_ROOT / "eitprocessing",
]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3resp import M3Session  # noqa: E402
from m3resp.adapters import EITProcessingAdapter  # noqa: E402
from m3resp.visualization.session import plot_eit_processing_summary  # noqa: E402


EIT_FILE = REPO_ROOT / "data" / "source" / "draeger_synthetic_draeger_20Hz.bin"
EIT_VENDOR = "draeger"  # supported: "draeger", "timpel", "sentec", "simulated"

EIT_SUBJECT_TYPE = "adult"
EIT_WELCH_WINDOW_SECONDS = 30.0
EIT_FILTER_MODE = "mdn"  # supported: "mdn", "lowpass", "bandpass", "none"
EIT_LOWPASS_HZ = 1.0
EIT_HIGHPASS_HZ = 0.05
EIT_FILTER_ORDER = 4
EIT_BREATH_MIN_DURATION_SECONDS = 2 / 3
EIT_COMPUTE_PIXEL_TIV = True

OUTPUT_DIR = REPO_ROOT / "output" / "eit-summary"


def preprocess_eit(
    sequence: Any,
    subject_type: str = EIT_SUBJECT_TYPE,
    welch_window_seconds: float = EIT_WELCH_WINDOW_SECONDS,
    filter_mode: str = EIT_FILTER_MODE,
    lowpass_hz: float = EIT_LOWPASS_HZ,
    highpass_hz: float = EIT_HIGHPASS_HZ,
    filter_order: int = EIT_FILTER_ORDER,
    breath_min_duration_seconds: float = EIT_BREATH_MIN_DURATION_SECONDS,
    compute_pixel_tiv: bool = EIT_COMPUTE_PIXEL_TIV,
) -> dict[str, Any]:
    """Run EIT rate detection, filtering, breath detection, TIV, and EELI."""

    import copy

    import numpy as np
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.rate_detection import RateDetection
    from eitprocessing.filters.butterworth_filters import ButterworthFilter
    from eitprocessing.filters.mdn import MDNFilter
    from eitprocessing.parameters.eeli import EELI
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    if not hasattr(sequence, "eit_data") or not hasattr(sequence, "continuous_data"):
        raise TypeError("preprocess_eit expects an eitprocessing Sequence.")
    if "raw" not in sequence.eit_data:
        raise KeyError("preprocess_eit requires sequence.eit_data['raw'].")
    if "global_impedance_(raw)" not in sequence.continuous_data:
        raise KeyError(
            "preprocess_eit requires "
            "sequence.continuous_data['global_impedance_(raw)']."
        )

    raw_eit = sequence.eit_data["raw"]
    raw_global_impedance = sequence.continuous_data["global_impedance_(raw)"]

    rate_detector = RateDetection(subject_type, welch_window=welch_window_seconds)
    rate_captures: dict[str, Any] = {}
    respiratory_rate_hz, heart_rate_hz = rate_detector.apply(
        raw_eit,
        captures=rate_captures,
        suppress_length_warnings=True,
        suppress_edge_case_warning=True,
    )

    filter_captures: dict[str, Any] = {}
    filtered_eit = raw_eit
    filtered_global_impedance = raw_global_impedance
    filter_mode = filter_mode.lower()

    if filter_mode == "mdn":
        eit_filter = MDNFilter(
            respiratory_rate=respiratory_rate_hz,
            heart_rate=heart_rate_hz,
        )
        filtered_eit = eit_filter.apply(
            raw_eit,
            captures=filter_captures,
            label="mdn_filtered",
            name="MDN-filtered EIT data",
            description="EIT data filtered with MDN heart-rate noise removal.",
        )
    elif filter_mode in {"lowpass", "bandpass"}:
        cutoff_frequency = (
            lowpass_hz if filter_mode == "lowpass" else (highpass_hz, lowpass_hz)
        )
        eit_filter = ButterworthFilter(
            filter_type=filter_mode,
            cutoff_frequency=cutoff_frequency,
            order=filter_order,
            sample_frequency=raw_eit.sample_frequency,
        )
        filtered_pixels = eit_filter.apply(
            np.nan_to_num(raw_eit.pixel_impedance),
            axis=0,
            captures=filter_captures,
        )
        filtered_eit = copy.deepcopy(raw_eit)
        filtered_eit.label = f"{filter_mode}_filtered"
        filtered_eit.name = f"{filter_mode.title()}-filtered EIT data"
        filtered_eit.description = (
            f"EIT data filtered with a {filter_mode} Butterworth filter."
        )
        filtered_eit.pixel_impedance = filtered_pixels
    elif filter_mode != "none":
        raise ValueError(
            "EIT_FILTER_MODE must be one of: 'mdn', 'lowpass', 'bandpass', "
            "'none'."
        )

    if filtered_eit is not raw_eit:
        sequence.eit_data.add(filtered_eit, overwrite=True)
        filtered_global_impedance = filtered_eit.get_summed_impedance(
            return_label=f"global_impedance_({filtered_eit.label})",
            name=f"Global impedance ({filtered_eit.label})",
            description="Global impedance calculated from filtered EIT data.",
        )
        sequence.continuous_data.add(filtered_global_impedance, overwrite=True)

    breath_detector = BreathDetection(minimum_duration=breath_min_duration_seconds)
    breath_intervals = breath_detector.find_breaths(
        filtered_global_impedance,
        result_label="eit_breaths",
        store=False,
    )
    sequence.interval_data.add(breath_intervals, overwrite=True)

    tiv_calculator = TIV(breath_detection=breath_detector)
    continuous_tiv = tiv_calculator.compute_parameter(
        filtered_global_impedance,
        sequence=sequence,
        store=False,
        result_label="continuous_tivs",
    )
    sequence.sparse_data.add(continuous_tiv, overwrite=True)

    eeli = EELI(breath_detection=breath_detector).compute_parameter(
        filtered_global_impedance,
        sequence=sequence,
        store=False,
        result_label="continuous_eelis",
    )
    sequence.sparse_data.add(eeli, overwrite=True)

    pixel_tiv = None
    if compute_pixel_tiv:
        pixel_tiv = tiv_calculator.compute_parameter(
            filtered_eit,
            filtered_global_impedance,
            sequence,
            tiv_timing="continuous",
            store=False,
            result_label="pixel_tivs",
        )
        sequence.sparse_data.add(pixel_tiv, overwrite=True)

    return {
        "sequence": sequence,
        "raw_eit": raw_eit,
        "raw_global_impedance": raw_global_impedance,
        "filtered_eit": filtered_eit,
        "filtered_global_impedance": filtered_global_impedance,
        "filter_mode": filter_mode,
        "filter_captures": filter_captures,
        "rate_detector": rate_detector,
        "rate_captures": rate_captures,
        "respiratory_rate_hz": respiratory_rate_hz,
        "heart_rate_hz": heart_rate_hz,
        "breath_intervals": breath_intervals,
        "continuous_tiv": continuous_tiv,
        "eeli": eeli,
        "pixel_tiv": pixel_tiv,
    }


def detect_eit_breaths(
    processed_eit: dict[str, Any], **kwargs: Any
) -> list[dict[str, Any]]:
    """Convert EITProcessing breath intervals to M3Resp breath rows."""

    events = []
    for breath in processed_eit["breath_intervals"].values:
        events.append(
            {
                "start_time": breath.start_time,
                "end_time": breath.end_time,
                "peak_time": breath.middle_time,
                "source": "eitprocessing.BreathDetection",
            }
        )
    return events


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

    session = M3Session(eit_adapter=EITProcessingAdapter())

    session.load_eit(EIT_FILE, vendor=EIT_VENDOR)
    session.preprocess_eit(preprocess=preprocess_eit)
    session.detect_eit_breaths(detector=detect_eit_breaths)
    session.export_summary(OUTPUT_DIR)

    return session, summarize_eit(session)


def save_figures(session: M3Session) -> None:
    """Save EIT processing figures."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eit_fig = plot_eit_processing_summary(session)
    eit_fig.savefig(OUTPUT_DIR / "eit-processing.png", dpi=150)

    rate_fig = session.processed["eit"]["rate_detector"].plotting.plot(
        **session.processed["eit"]["rate_captures"]
    )
    rate_fig.savefig(OUTPUT_DIR / "eit-rate-detection.png", dpi=150)


def main() -> None:
    """Run the example and print a compact EIT summary."""

    session, eit_summary = run_workflow()
    save_figures(session)

    print("EIT workflow complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    for key, value in eit_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
