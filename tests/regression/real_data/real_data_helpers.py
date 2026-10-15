"""Shared code for the real-data tests and for
`scripts/update_real_data_references.py`.

Reads the dataset list (`datasets.yaml`), finds and checks the recordings,
lines the devices up on the Biopac clock by cross-correlating the airway
pressure they all record, runs the m3resp workflows on each analysed window,
finds breath onsets in the FluxMed flow, and collects every number the tests
compare (`collect_results`).

Times are in seconds on the Biopac clock unless a name says otherwise. Rates
are per minute.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import os
import warnings
from typing import Any

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATASETS_FILE = os.path.join(HERE, "datasets.yaml")
REFERENCE_FILE = os.path.join(HERE, "reference_values.json")

#: Environment variable naming the folder that holds the recordings.
DATA_FOLDER_VARIABLE = "M3RESP_REAL_DATA"
DEFAULT_DATA_FOLDER = os.path.join(REPO_ROOT, "data", "source", "Test_PS_27072026")

#: Settings shared by every EMG chain (those of the paper and of the
#: recording1_a2 example workflow).
EMG_SAMPLE_FREQUENCY = 2000.0
EMG_SETTINGS: dict[str, Any] = {
    "high_pass_hz": 20.0,
    "low_pass_hz": 500.0,
    "notch_base_frequency": 50.0,
    "notch_quality_factor": 30.0,
    "notch_max_frequency": 950.0,
    "ecg_peak_fraction": 0.2,
    "wavelet_levels": 4,
    "wavelet_threshold": 4.5,
    "envelope_window_seconds": 0.85,
    "envelope_method": "median",
    "baseline_window_seconds": 7.5,
    "baseline_step_seconds": 0.2,
    "baseline_percentile": 33.0,
    "min_breath_width_seconds": 0.7,
    "prominence_factor": 0.4,
}

#: The EMG filter chains compared with each other. Each is (notch filter
#: used, ECG removal: "none", "gating" or "wavelet").
EMG_CHAINS: dict[str, tuple[bool, str]] = {
    "bandpass": (False, "none"),
    "notch_bandpass": (True, "none"),
    "notch_bandpass_gating": (True, "gating"),
    "notch_bandpass_wavelet": (True, "wavelet"),
}
#: The chain the paper uses; its results are the main EMG results.
MAIN_EMG_CHAIN = "notch_bandpass_wavelet"

#: Respiratory band for the flow signal, in Hz (as in the paper).
FLOW_BAND_HZ = (0.08, 1.2)
#: Two breath onsets count as the same breath when they are at most this far
#: apart, in seconds (as in the paper).
MATCH_TOLERANCE_SECONDS = 1.0

#: Lining up the devices by their airway pressure: both pressures are
#: low-passed at SYNC_LOWPASS_HZ, the Biopac pressure is taken from
#: SYNC_MARGIN_SECONDS before to SYNC_MARGIN_SECONDS after the window, and
#: the time shift with the highest correlation is searched within
#: +-SYNC_SEARCH_SECONDS, first in steps of 1 / SYNC_COARSE_RATE_HZ, then in
#: steps of SYNC_FINE_STEP_SECONDS around the best coarse shift.
SYNC_LOWPASS_HZ = 8.0
SYNC_MARGIN_SECONDS = 20.0
SYNC_SEARCH_SECONDS = 30.0
SYNC_COARSE_RATE_HZ = 100.0
SYNC_FINE_STEP_SECONDS = 0.001


def load_datasets() -> dict[str, Any]:
    """Return the contents of `datasets.yaml`."""

    with open(DATASETS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def data_folder() -> str:
    """Folder holding the recordings: `M3RESP_REAL_DATA`, or the default."""

    folder = os.environ.get(DATA_FOLDER_VARIABLE) or DEFAULT_DATA_FOLDER
    return os.path.abspath(os.path.expanduser(folder))


@functools.cache
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_paths(name: str) -> tuple[dict[str, str] | None, str]:
    """Absolute paths of a dataset's files, after checking their checksums.

    Returns:
        (paths, reason): `paths` maps each file role (``biopac``,
        ``fluxmed``, ``eit``) to its path, or is None when a file is missing;
        `reason` then says which file.

    Raises:
        ValueError: If a file exists but its SHA-256 checksum differs from
            the one in `datasets.yaml`.
    """

    folder = data_folder()
    paths: dict[str, str] = {}
    for role, entry in load_datasets()[name]["files"].items():
        path = os.path.join(folder, *entry["path"].split("/"))
        if not os.path.isfile(path):
            return None, f"{name}: {path} not found (set {DATA_FOLDER_VARIABLE})"
        if _sha256(path) != entry["sha256"]:
            raise ValueError(
                f"{path} differs from the file the reference values were made "
                f"from (SHA-256 {entry['sha256']})."
            )
        paths[role] = path
    return paths, ""


# --- Reading the files -------------------------------------------------------


def read_biopac(path: str) -> tuple[float, np.ndarray]:
    """Read a Biopac export with m3resp's EMG loader.

    Returns:
        (sample_frequency, array): Hz, and the channels (rows) by samples.
    """

    from m3resp import M3Session

    session = M3Session()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load_emg(path)
    assert session.emg is not None
    metadata = session.emg.data["metadata"]
    return float(metadata["fs"]), np.asarray(session.emg.data["array"], dtype=float)


def read_fluxmed(path: str) -> dict[str, Any]:
    """Read a FluxMed "Signals.txt" export.

    The file is tab-separated with comma decimal marks: five lines of device
    information (one gives the sampling rate), a line of column names and a
    line of units, then one row per sample: time (s), flow (l/min), volume
    (ml), airway pressure (cmH2O), and more pressures.

    Returns:
        dict: ``sample_frequency`` (Hz), ``flow`` (l/min), ``volume`` (ml)
        and ``airway_pressure`` (cmH2O).
    """

    import pandas as pd

    with open(path, encoding="latin-1") as handle:
        header = [handle.readline() for _ in range(5)]
    rate_line = next(
        line for line in header if line.lower().startswith("sampling rate")
    )
    table = pd.read_csv(
        path,
        sep="\t",
        skiprows=7,
        header=None,
        usecols=[1, 2, 3],
        names=["flow", "volume", "airway_pressure"],
        decimal=",",
        encoding="latin-1",
    ).dropna()
    return {
        "sample_frequency": float(rate_line.split(":")[1].split()[0]),
        **{column: table[column].to_numpy(dtype=float) for column in table},
    }


def read_eit_airway_pressure(path: str, channel: str) -> dict[str, Any]:
    """Airway pressure recorded in a Draeger EIT file, read with eitprocessing.

    Returns:
        dict: ``sample_frequency`` (Hz), ``values`` (cmH2O) and
        ``first_frame_time`` (the file's own time of its first frame, s).
    """

    from eitprocessing.datahandling.loading import load_eit_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequence = load_eit_data(path, vendor="draeger")
    pressure = sequence.continuous_data[channel]
    return {
        "sample_frequency": float(sequence.eit_data["raw"].sample_frequency),
        "values": np.asarray(pressure.values, dtype=float),
        "first_frame_time": float(sequence.eit_data["raw"].time[0]),
    }


# --- Lining up the devices ---------------------------------------------------


def _lowpass(values: np.ndarray, sample_frequency: float) -> np.ndarray:
    """Second-order Butterworth low-pass at SYNC_LOWPASS_HZ, forward and back."""

    from scipy.signal import butter, filtfilt

    b, a = butter(2, SYNC_LOWPASS_HZ / (sample_frequency / 2), "low")
    return filtfilt(b, a, values)


def _without_missing_end(values: np.ndarray) -> np.ndarray:
    """Drop missing samples (NaN) at the end; a NaN elsewhere is an error."""

    finite = np.isfinite(values)
    last = int(np.flatnonzero(finite)[-1]) + 1
    if not finite[:last].all():
        raise ValueError("airway pressure has missing samples inside the recording")
    return values[:last]


def _overlap_correlations(
    segment: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation coefficient of `segment` with `other` at every whole-sample
    shift, over the samples where the two overlap.

    Returns:
        (coefficients, overlap): for shift k (index into `other` where the
        segment starts; k runs from -(len(segment) - 1) to len(other) - 1),
        the correlation coefficient and the number of overlapping samples.
    """

    from scipy.signal import correlate

    def slide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return correlate(a, b, mode="full", method="fft")

    ones_segment = np.ones_like(segment)
    ones_other = np.ones_like(other)
    overlap = np.rint(slide(ones_other, ones_segment))
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_s = slide(ones_other, segment)
        sum_ss = slide(ones_other, segment**2)
        sum_o = slide(other, ones_segment)
        sum_oo = slide(other**2, ones_segment)
        sum_so = slide(other, segment)
        covariance = sum_so - sum_s * sum_o / overlap
        spread = (sum_ss - sum_s**2 / overlap) * (sum_oo - sum_o**2 / overlap)
        coefficients = covariance / np.sqrt(spread)
    return coefficients, overlap


def paw_offset(
    biopac_pressure: np.ndarray,
    biopac_fs: float,
    other_pressure: np.ndarray,
    other_fs: float,
    *,
    window: tuple[float, float],
) -> dict[str, float]:
    """Where another device's first sample lies on the Biopac clock.

    Both devices record the same airway pressure, so the shift that makes the
    two pressure traces most alike (highest correlation coefficient) lines
    the clocks up. Only the Biopac pressure around `window` is used, so the
    shift holds for that window even when one clock runs slightly fast. A
    shift counts only when the other recording covers the whole window; the
    correlation is taken over the part of the stretch both recordings cover.

    Returns:
        dict: ``offset`` (s, Biopac time of the other device's first sample)
        and ``correlation`` (correlation coefficient at that offset).
    """

    reference = _lowpass(_without_missing_end(biopac_pressure), biopac_fs)
    other = _lowpass(_without_missing_end(other_pressure), other_fs)
    reference_time = np.arange(len(reference)) / biopac_fs
    other_time = np.arange(len(other)) / other_fs

    start = max(0.0, window[0] - SYNC_MARGIN_SECONDS)
    end = min(reference_time[-1], window[1] + SYNC_MARGIN_SECONDS)
    step = 1.0 / SYNC_COARSE_RATE_HZ
    segment_time = start + np.arange(int((end - start) * SYNC_COARSE_RATE_HZ)) * step
    segment = np.interp(segment_time, reference_time, reference)
    other_grid = np.interp(
        np.arange(int(other_time[-1] * SYNC_COARSE_RATE_HZ) + 1) * step,
        other_time,
        other,
    )

    # Coarse: every shift of a whole step.
    coefficients, overlap = _overlap_correlations(segment, other_grid)
    shifts = np.arange(len(coefficients)) - (len(segment) - 1)
    offsets = start - shifts * step
    covers_window = (offsets <= window[0]) & (offsets + other_time[-1] >= window[1])
    allowed = (np.abs(offsets) <= SYNC_SEARCH_SECONDS) & covers_window & (overlap > 1)
    if not allowed.any():
        raise ValueError("the other recording does not cover the window")
    best = float(offsets[allowed][int(np.nanargmax(coefficients[allowed]))])

    # Fine: steps of SYNC_FINE_STEP_SECONDS around the best coarse shift.
    def coefficient(offset: float) -> float:
        shifted = segment_time - offset
        inside = (shifted >= other_time[0]) & (shifted <= other_time[-1])
        values = np.interp(shifted[inside], other_time, other)
        return float(np.corrcoef(segment[inside], values)[0, 1])

    candidates = best + SYNC_FINE_STEP_SECONDS * np.arange(-20, 21)
    fine = [coefficient(offset) for offset in candidates]
    index = int(np.argmax(fine))
    return {
        "offset": round(float(candidates[index]), 6),
        "correlation": float(fine[index]),
    }


# --- Workflows ---------------------------------------------------------------


def eit_spec(
    eit_path: str, window: tuple[float, float], eit_offset: float
) -> dict[str, Any]:
    """Workflow for the EIT part: cut to the window, rates, MDN filter,
    global impedance, breaths."""

    return {
        "name": "real_data_eit",
        "inputs": {"eit_file": eit_path},
        "steps": [
            {"uses": "eit.load", "with": {"file_path": "@eit_file", "vendor": "draeger"}},
            {
                "uses": "eit.slice_recording",
                "with": {
                    "start_seconds": window[0] - eit_offset,
                    "end_seconds": window[1] - eit_offset,
                },
            },
            {"uses": "eit.detect_rates", "in": {"signal": "raw_eit"}, "with": {"subject_type": "adult"}},
            {"uses": "eit.mdn_filter", "in": {"signal": "raw_eit"}},
            {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
            {"uses": "eit.detect_breaths", "in": {"signal": "global_impedance"}},
        ],
    }  # fmt: skip


def emg_spec(
    biopac_path: str,
    window: tuple[float, float],
    channel: int,
    *,
    notch: bool,
    ecg_removal: str,
) -> dict[str, Any]:
    """Workflow for one EMG filter chain on one window."""

    s = EMG_SETTINGS
    preprocess: dict[str, Any] = {
        "channel": channel,
        "high_pass_hz": s["high_pass_hz"],
        "low_pass_hz": s["low_pass_hz"],
    }
    if notch:
        preprocess.update(
            notch_base_frequency=s["notch_base_frequency"],
            notch_quality_factor=s["notch_quality_factor"],
            notch_max_frequency=s["notch_max_frequency"],
            notch_before_bandpass=True,
        )
    steps: list[dict[str, Any]] = [
        {"uses": "emg.load", "with": {"file_path": "@emg_file"}},
        {"uses": "emg.slice_recording", "with": {"start_seconds": window[0], "end_seconds": window[1]}},
        {"uses": "emg.preprocess", "with": preprocess, "out": {"processed_emg": "filtered_emg"}},
        {
            "uses": "emg.ecg_detect_peaks",
            "in": {"processed_emg": "filtered_emg"},
            "with": {"source": "filtered", "peak_fraction": s["ecg_peak_fraction"]},
        },
    ]  # fmt: skip
    if ecg_removal == "wavelet":
        steps.append(
            {
                "uses": "emg.ecg_wavelet_denoising",
                "in": {"processed_emg": "filtered_emg"},
                "with": {
                    "source": "filtered",
                    "levels": s["wavelet_levels"],
                    "fixed_threshold": s["wavelet_threshold"],
                    "hard_thresholding": True,
                },
            }
        )
    elif ecg_removal == "gating":
        steps.append(
            {
                "uses": "emg.ecg_gating",
                "in": {"processed_emg": "filtered_emg"},
                "with": {"source": "filtered"},
            }
        )
    envelope_input = (
        "filtered_emg" if ecg_removal == "none" else "processed_emg_after_ecg"
    )
    steps += [
        {
            "uses": "emg.envelope",
            "in": {"processed_emg": envelope_input},
            "with": {"window_seconds": s["envelope_window_seconds"], "method": s["envelope_method"]},
            "out": {"processed_emg_with_envelope": "processed_emg"},
        },
        {
            "uses": "emg.moving_baseline",
            "with": {
                "window_seconds": s["baseline_window_seconds"],
                "step_seconds": s["baseline_step_seconds"],
                "percentile": s["baseline_percentile"],
            },
        },
        {
            "uses": "emg.detect_breaths",
            "with": {
                "min_breath_width_seconds": s["min_breath_width_seconds"],
                "prominence_factor": s["prominence_factor"],
                "merge_close_peaks_within_width": True,
            },
        },
        {"uses": "emg.peak_indices"},
        {"uses": "emg.onoffpeak_baseline_crossing"},
        {"uses": "emg.respiratory_rate"},
    ]  # fmt: skip
    return {
        "name": "real_data_emg",
        "inputs": {"emg_file": biopac_path},
        "steps": steps,
    }


def run_spec(spec: dict[str, Any]) -> tuple[Any, Any]:
    """Run a workflow spec in a new session.

    Returns:
        (context, session): the workflow's named results and the session.
    """

    from m3resp import M3Session
    from m3resp.workflows import load_spec, run_workflow

    session = M3Session()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_workflow(load_spec(spec, root=REPO_ROOT), session=session)
    return result.context, session


# --- Flow reference ----------------------------------------------------------


def flow_onsets(flow: np.ndarray, sample_frequency: float) -> np.ndarray:
    """Sample index of the start of each inspiration in the flow.

    The flow is band-passed to the respiratory band (0.08-1.2 Hz, second
    order, forward and backward). Each inspiration is a peak of the filtered
    flow, at most one per breath period; its start is the first sample after
    the filtered flow last crossed zero before the peak.
    """

    from scipy.signal import butter, filtfilt, find_peaks, welch

    nyquist = sample_frequency / 2
    b, a = butter(2, [FLOW_BAND_HZ[0] / nyquist, FLOW_BAND_HZ[1] / nyquist], "band")
    filtered = filtfilt(b, a, np.asarray(flow, dtype=float))

    segment = min(len(filtered), int(30 * sample_frequency))
    frequencies, power = welch(filtered, fs=sample_frequency, nperseg=segment)
    band = (frequencies >= 0.1) & (frequencies <= 0.8)
    period_seconds = 1.0 / frequencies[band][int(np.argmax(power[band]))]

    swing = np.percentile(filtered, 99) - np.percentile(filtered, 1)
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.6 * period_seconds * sample_frequency),
        prominence=0.3 * swing,
    )
    onsets = []
    for peak in peaks:
        at_or_below_zero = np.flatnonzero(filtered[:peak] <= 0)
        if len(at_or_below_zero):
            onsets.append(int(at_or_below_zero[-1]) + 1)
    return np.asarray(onsets, dtype=int)


def match_events(
    detected: np.ndarray, reference: np.ndarray, tolerance: float
) -> tuple[int, int, int]:
    """Pair events one to one, closest pairs first, within `tolerance`.

    Returns:
        (matched, extra, missed): pairs found, detected events without a
        reference event, reference events without a detected event.
    """

    candidates = sorted(
        (abs(d - r), i, j)
        for i, d in enumerate(detected)
        for j, r in enumerate(reference)
        if abs(d - r) <= tolerance
    )
    used_detected: set[int] = set()
    used_reference: set[int] = set()
    for _, i, j in candidates:
        if i not in used_detected and j not in used_reference:
            used_detected.add(i)
            used_reference.add(j)
    matched = len(used_detected)
    return matched, len(detected) - matched, len(reference) - matched


def f1_score(matched: int, extra: int, missed: int) -> float:
    """F1 = 2 matched / (2 matched + extra + missed)."""

    return 2 * matched / (2 * matched + extra + missed)


def rate_from_times(times: np.ndarray) -> float:
    """Events per minute from the mean interval between event times (s)."""

    return float(60.0 / np.mean(np.diff(times)))


# --- All results of one dataset ---------------------------------------------


def run_dataset(name: str, paths: dict[str, str]) -> dict[str, Any]:
    """Line up the devices and run every workflow of a dataset.

    Returns a dict with:

    - ``dataset``: the settings from `datasets.yaml`;
    - ``sync``: window -> device -> ``offset`` and ``correlation`` (see
      `paw_offset`);
    - ``eit_context``, ``eit_session``: the EIT workflow on the EIT window;
    - ``eit_first_frame_time``: the EIT file's own time of its first frame;
    - ``emg``: window -> filter chain -> workflow context;
    - ``fluxmed``: the FluxMed recording (see `read_fluxmed`).
    """

    dataset = load_datasets()[name]
    pressure = dataset["airway_pressure"]
    biopac_fs, biopac = read_biopac(paths["biopac"])
    biopac_pressure = biopac[pressure["biopac_channel"]]
    fluxmed = read_fluxmed(paths["fluxmed"])
    eit_pressure = read_eit_airway_pressure(paths["eit"], pressure["eit_channel"])

    sync: dict[str, dict[str, dict[str, float]]] = {}
    for window_name, window in dataset["windows"].items():
        sync[window_name] = {
            "fluxmed": paw_offset(
                biopac_pressure,
                biopac_fs,
                fluxmed["airway_pressure"],
                fluxmed["sample_frequency"],
                window=window,
            )
        }
    eit_window = dataset["eit_window"]
    sync[eit_window]["eit"] = paw_offset(
        biopac_pressure,
        biopac_fs,
        eit_pressure["values"],
        eit_pressure["sample_frequency"],
        window=dataset["windows"][eit_window],
    )

    eit_context, eit_session = run_spec(
        eit_spec(
            paths["eit"],
            dataset["windows"][eit_window],
            sync[eit_window]["eit"]["offset"],
        )
    )
    emg = {
        window_name: {
            chain: run_spec(
                emg_spec(
                    paths["biopac"],
                    window,
                    dataset["emgdi_channel"],
                    notch=notch,
                    ecg_removal=removal,
                )
            )[0]
            for chain, (notch, removal) in EMG_CHAINS.items()
        }
        for window_name, window in dataset["windows"].items()
    }
    return {
        "dataset": dataset,
        "sync": sync,
        "eit_context": eit_context,
        "eit_session": eit_session,
        "eit_first_frame_time": eit_pressure["first_frame_time"],
        "emg": emg,
        "fluxmed": fluxmed,
    }


def flow_onset_times(run: dict[str, Any], window_name: str) -> np.ndarray:
    """Flow onsets inside a window, on the Biopac clock."""

    fluxmed = run["fluxmed"]
    start, end = run["dataset"]["windows"][window_name]
    times = (
        flow_onsets(fluxmed["flow"], fluxmed["sample_frequency"])
        / fluxmed["sample_frequency"]
        + run["sync"][window_name]["fluxmed"]["offset"]
    )
    return times[(times >= start) & (times < end)]


def eit_breath_start_times(run: dict[str, Any]) -> np.ndarray:
    """Start of each EIT breath (end of expiration), on the Biopac clock."""

    eit_window = run["dataset"]["eit_window"]
    starts = np.asarray(
        [
            breath.start_time
            for breath in run["eit_context"].get("breath_intervals").values
        ],
        dtype=float,
    )
    return (
        starts - run["eit_first_frame_time"] + run["sync"][eit_window]["eit"]["offset"]
    )


def _emg_results(
    context: Any, window: tuple[float, float], flow_times: np.ndarray
) -> dict[str, Any]:
    fs = EMG_SAMPLE_FREQUENCY
    ecg_peaks = np.asarray(context.get("ecg_peak_indices"), dtype=int)
    peaks = np.asarray(context.get("peak_indices"), dtype=int)
    onsets = np.asarray(context.get("start_indices"), dtype=int)
    envelope = np.asarray(context.get("processed_emg")["envelope"], dtype=float)
    matched, extra, missed = match_events(
        onsets / fs + window[0], flow_times, MATCH_TOLERANCE_SECONDS
    )
    return {
        "ecg_peak_indices": ecg_peaks.tolist(),
        "heart_rate": float(60.0 / np.median(np.diff(ecg_peaks) / fs)),
        "breath_peak_indices": peaks.tolist(),
        "breath_onset_indices": onsets.tolist(),
        "respiratory_rate": float(context.get("respiratory_rate")[0]),
        "flow_match": {"matched": matched, "extra": extra, "missed": missed},
        # Area under the envelope divided by the window length (mV).
        "envelope_area_per_second": float(
            np.trapezoid(envelope, dx=1.0 / fs) / (len(envelope) / fs)
        ),
        "baseline_sum": float(np.sum(context.get("baseline"))),
    }


def collect_results(run: dict[str, Any]) -> dict[str, Any]:
    """Every number the tests check, as plain numbers and lists.

    EMG sample indices are counted from the start of their window. Rates are
    per minute; times in seconds on the Biopac clock.
    """

    dataset = run["dataset"]
    windows = dataset["windows"]
    eit = run["eit_context"]
    eit_breaths = eit_breath_start_times(run)
    results: dict[str, Any] = {
        "sync": run["sync"],
        "flow": {},
        "eit": {
            "kept_frames": int(
                run["eit_session"].parameters["eit_slice"]["kept_frames"]
            ),
            "respiratory_rate": float(eit.get("respiratory_rate_hz")) * 60.0,
            "heart_rate": float(eit.get("heart_rate_hz")) * 60.0,
            "breath_start_times": [round(float(t), 6) for t in eit_breaths],
        },
        "emg": {},
    }
    for window_name, window in windows.items():
        flow_times = flow_onset_times(run, window_name)
        results["flow"][window_name] = {
            "onset_times": [round(float(t), 6) for t in flow_times],
            "respiratory_rate": rate_from_times(flow_times),
        }
        results["emg"][window_name] = {
            chain: _emg_results(context, window, flow_times)
            for chain, context in run["emg"][window_name].items()
        }
    return results


def write_reference_values(results: dict[str, dict[str, Any]]) -> None:
    """Write `reference_values.json` (dataset name -> collected results)."""

    with open(REFERENCE_FILE, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, sort_keys=True)
        handle.write("\n")


def read_reference_values() -> dict[str, Any]:
    """Read `reference_values.json`, or an empty dict when it does not exist."""

    if not os.path.isfile(REFERENCE_FILE):
        return {}
    with open(REFERENCE_FILE, encoding="utf-8") as handle:
        return json.load(handle)
