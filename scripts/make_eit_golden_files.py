"""Save the eitprocessing results for the EIT test files as golden files.

The script reads each EIT test file with eitprocessing and runs the EIT
chain on it: rate detection, MDN filter, global impedance, breath detection,
TIV, EELI, pixel TIV, pixel breaths and the lung-space masks. It saves every
result as numbers in ``.npz`` files, one folder per test file, and writes a
``manifest.json`` with the eitprocessing version and commit, the checksum of
each input file and the settings. The comparison tests read these files, so
they run without eitprocessing.

Run it with the reference version of eitprocessing (1.8.7) installed.

Usage: ``python scripts/make_eit_golden_files.py [--output FOLDER]``
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")

REFERENCE_VERSION = "1.8.7"

# Test file name, vendor, and the folder name of its golden files.
INPUTS = (
    ("draeger_synthetic_draeger_20Hz.bin", "draeger", "draeger_synthetic_20Hz"),
    ("sentec_healthy_volunteer_1b.zri", "sentec", "sentec_healthy_volunteer_1b"),
    (
        "timpel_healthy_volunteer_1_60-120s.txt",
        "timpel",
        "timpel_healthy_volunteer_1_60-120s",
    ),
)

# The settings of the chain, the same as in
# tests/regression/test_eit_gap_migration_equivalence.py.
SETTINGS: dict[str, Any] = {
    "rate_detection": {"subject_type": "adult"},
    "breath_detection": {"minimum_duration_s": 2 / 3},
    "pixel_tiv": {"tiv_timing": "continuous"},
    "pixel_breath": {"phase_correction_mode": "negative amplitude"},
    "tiv_lungspace": {"threshold": 0.15},
    "amplitude_lungspace": {"threshold": 0.15},
    "watershed_lungspace": {"threshold_fraction": 0.15},
    "filter_by_size": {"min_region_size": 1},
}

# Time windows (seconds after the first frame) for the slicing results, and
# the (start inclusive, end inclusive) flags tried on each window.
SLICE_WINDOWS = ((10.0, 20.0), (10.0, 10.0), (0.0, 1.0))
SLICE_FLAGS = ((False, False), (True, False), (False, True), (True, True))

COMPARISON_RULE = (
    "Shapes, counts, indices and masks exact. Other floating-point values may "
    "differ by at most 1e-9 times the largest absolute value of the same "
    "array (NaN must stay NaN)."
)


def sha256_of_file(path: str) -> str:
    """Return the SHA-256 checksum of a file.

    Args:
        path (str): The file to read.

    Returns:
        str: The checksum as hexadecimal text.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eitprocessing_commit() -> str | None:
    """Return the git commit eitprocessing was installed from.

    Returns:
        str | None: The commit hash, or None when pip stored no commit
        (an install from a package index).
    """

    distribution = importlib.metadata.distribution("eitprocessing")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def breath_array(breaths: Any) -> np.ndarray:
    """Return start, middle and end time of each breath as an (n, 3) array.

    Args:
        breaths (list): eitprocessing ``Breath`` objects.

    Returns:
        np.ndarray: One row per breath, times in seconds.
    """

    rows = [(b.start_time, b.middle_time, b.end_time) for b in breaths]
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def map_values(item: Any) -> np.ndarray | None:
    """Return the numbers of a pixel map, a pixel mask, an array or a number.

    Args:
        item (Any): An eitprocessing ``PixelMap`` or ``PixelMask``, a NumPy
            array or a number.

    Returns:
        np.ndarray | None: The numbers, or None for an item of another type.
    """

    if isinstance(item, np.ndarray | int | float | np.number):
        return np.asarray(item)
    if isinstance(getattr(item, "mask", None), np.ndarray):
        return np.asarray(item.mask)
    if isinstance(getattr(item, "values", None), np.ndarray):
        return np.asarray(item.values)
    return None


def add_captures(
    arrays: dict[str, np.ndarray], prefix: str, captures: dict
) -> list[str]:
    """Copy the numbers of an eitprocessing captures dictionary into ``arrays``.

    Args:
        arrays (dict): The arrays of one ``.npz`` file. New entries are added.
        prefix (str): Text put before each capture name.
        captures (dict): The captures of one eitprocessing call.

    Returns:
        list[str]: Names of the captures that are left out: copies of whole
        EIT data, and items of other types. The manifest lists them.
    """

    left_out = []
    for name, item in captures.items():
        if hasattr(item, "pixel_impedance"):
            left_out.append(name)
            continue
        if hasattr(item, "time") and hasattr(item, "values"):
            arrays[f"{prefix}{name}/time"] = np.asarray(item.time, dtype=float)
            arrays[f"{prefix}{name}/values"] = np.asarray(item.values, dtype=float)
            continue
        if isinstance(item, list | tuple) and all(isinstance(x, tuple) for x in item):
            arrays[f"{prefix}{name}"] = np.asarray(item, dtype=float)
            continue
        values = map_values(item)
        if values is None:
            left_out.append(name)
        else:
            arrays[f"{prefix}{name}"] = values
    return left_out


def reader_arrays(sequence: Any, eit_data: Any) -> dict[str, np.ndarray]:
    """Collect everything the reader returns as arrays.

    Args:
        sequence (Any): The eitprocessing ``Sequence`` of one file.
        eit_data (Any): The raw ``EITData`` in that sequence.

    Returns:
        dict: Arrays for ``reader.npz``.
    """

    arrays: dict[str, np.ndarray] = {
        "time": np.asarray(eit_data.time, dtype=float),
        "sample_frequency": np.asarray(eit_data.sample_frequency, dtype=float),
        "pixel_impedance": np.asarray(eit_data.pixel_impedance, dtype=float),
        "vendor": np.asarray(eit_data.vendor.value),
    }
    for label, data in sequence.continuous_data.items():
        arrays[f"continuous/{label}/time"] = np.asarray(data.time, dtype=float)
        arrays[f"continuous/{label}/values"] = np.asarray(data.values)
        arrays[f"continuous/{label}/unit"] = np.asarray(str(data.unit))
    for label, data in sequence.sparse_data.items():
        arrays[f"sparse/{label}/time"] = np.asarray(data.time, dtype=float)
        if data.values is not None:
            arrays[f"sparse/{label}/marker"] = np.asarray(
                [e.marker for e in data.values]
            )
            arrays[f"sparse/{label}/text"] = np.asarray([e.text for e in data.values])
    for label, data in sequence.interval_data.items():
        arrays[f"interval/{label}/start_end"] = np.asarray(
            [(i.start_time, i.end_time) for i in data.intervals], dtype=float
        ).reshape(-1, 2)
        if data.values:
            arrays[f"interval/{label}/breaths"] = breath_array(data.values)
    return arrays


def slicing_arrays(eit_data: Any) -> dict[str, np.ndarray]:
    """Cut the raw EIT data at fixed times and keep which frames remain.

    The cuts start and end on frame times, so the inclusive flags change the
    result.

    Args:
        eit_data (Any): The raw ``EITData`` of one file.

    Returns:
        dict: For each window and flag pair, the array
        ``[index of first frame, index of last frame, number of frames]``
        (indices -1 when no frame remains).
    """

    arrays = {}
    time = np.asarray(eit_data.time, dtype=float)
    for start, end in SLICE_WINDOWS:
        start_time = time[np.searchsorted(time, time[0] + start)]
        end_time = time[np.searchsorted(time, time[0] + end)]
        for start_inclusive, end_inclusive in SLICE_FLAGS:
            part = eit_data.select_by_time(
                start_time,
                end_time,
                start_inclusive=start_inclusive,
                end_inclusive=end_inclusive,
            )
            frames = len(part.time)
            first = int(np.searchsorted(time, part.time[0])) if frames else -1
            last = int(np.searchsorted(time, part.time[-1])) if frames else -1
            key = f"select_by_time/{start}-{end}/{int(start_inclusive)}{int(end_inclusive)}"
            arrays[key] = np.asarray([first, last, frames])
    return arrays


def run_chain(
    path: str, vendor: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Run the eitprocessing chain on one file.

    Args:
        path (str): The EIT file.
        vendor (str): ``"draeger"``, ``"sentec"`` or ``"timpel"``.

    Returns:
        tuple: The arrays per ``.npz`` file name, and notes for the manifest
        (the warnings of each step and the captures left out).
    """

    from eitprocessing.datahandling.loading import load_eit_data
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.pixel_breath import PixelBreath
    from eitprocessing.features.rate_detection import RateDetection
    from eitprocessing.filters.mdn import MDNFilter
    from eitprocessing.parameters.eeli import EELI
    from eitprocessing.parameters.tidal_impedance_variation import TIV
    from eitprocessing.roi.amplitude import AmplitudeLungspace
    from eitprocessing.roi.filter_by_size import FilterROIBySize
    from eitprocessing.roi.tiv import TIVLungspace
    from eitprocessing.roi.watershed import WatershedLungspace

    notes: dict[str, Any] = {"warnings": {}, "captures_left_out": {}}
    files: dict[str, dict[str, np.ndarray]] = {}

    def record(step: str, caught: list) -> None:
        messages = sorted({str(w.message) for w in caught})
        notes["warnings"][step] = {"count": len(caught), "messages": messages}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sequence = load_eit_data(path, vendor=vendor)
    record("reader", caught)
    raw = next(iter(sequence.eit_data.values()))
    files["reader"] = reader_arrays(sequence, raw)
    files["slicing"] = slicing_arrays(raw)

    rate_captures: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        respiratory_rate, heart_rate = RateDetection(
            SETTINGS["rate_detection"]["subject_type"]
        ).apply(
            raw,
            captures=rate_captures,
            suppress_length_warnings=True,
            suppress_edge_case_warning=True,
        )
    record("rates", caught)
    files["rates"] = {
        "respiratory_rate_hz": np.asarray(respiratory_rate, dtype=float),
        "heart_rate_hz": np.asarray(heart_rate, dtype=float),
    }
    notes["captures_left_out"]["rates"] = add_captures(
        files["rates"], "captures/", rate_captures
    )

    mdn_captures: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filtered = MDNFilter(
            respiratory_rate=respiratory_rate, heart_rate=heart_rate
        ).apply(raw, captures=mdn_captures, label="mdn_filtered")
    record("mdn_filter", caught)
    files["mdn_filter"] = {
        "pixel_impedance": np.asarray(filtered.pixel_impedance, dtype=float)
    }
    notes["captures_left_out"]["mdn_filter"] = add_captures(
        files["mdn_filter"], "captures/", mdn_captures
    )

    raw_global = raw.get_summed_impedance(return_label="global_impedance_(raw)")
    global_impedance = filtered.get_summed_impedance(
        return_label="global_impedance_(mdn_filtered)"
    )
    files["global_impedance"] = {
        "raw/time": np.asarray(raw_global.time, dtype=float),
        "raw/values": np.asarray(raw_global.values, dtype=float),
        "filtered/time": np.asarray(global_impedance.time, dtype=float),
        "filtered/values": np.asarray(global_impedance.values, dtype=float),
    }

    minimum_duration = SETTINGS["breath_detection"]["minimum_duration_s"]
    breath_detector = BreathDetection(minimum_duration=minimum_duration)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        breaths = breath_detector.find_breaths(global_impedance)
    record("breaths", caught)
    files["breaths"] = {"breaths": breath_array(breaths.values)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        continuous_tiv: Any = TIV(breath_detection=breath_detector).compute_parameter(
            global_impedance, store=False, result_label="continuous_tivs"
        )
        eeli = EELI(breath_detection=breath_detector).compute_parameter(
            global_impedance, store=False, result_label="continuous_eelis"
        )
    record("tiv_eeli", caught)
    files["tiv_eeli"] = {
        "tiv/time": np.asarray(continuous_tiv.time, dtype=float),
        "tiv/values": np.asarray(continuous_tiv.values, dtype=float),
        "eeli/time": np.asarray(eeli.time, dtype=float),
        "eeli/values": np.asarray(eeli.values, dtype=float),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pixel_tiv = TIV(breath_detection=breath_detector).compute_parameter(
            filtered,
            global_impedance,
            None,
            tiv_timing=SETTINGS["pixel_tiv"]["tiv_timing"],
            store=False,
            result_label="pixel_tivs",
        )
    record("pixel_tiv", caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pixel_breaths = PixelBreath(
            breath_detection=BreathDetection(minimum_duration=minimum_duration),
            phase_correction_mode=SETTINGS["pixel_breath"]["phase_correction_mode"],
        ).find_pixel_breaths(filtered, global_impedance, store=False)
    record("pixel_breaths", caught)
    # One Breath (or None) per breath and pixel; keep its three times.
    per_pixel = np.asarray(pixel_breaths.values, dtype=object)
    landmarks = np.full(per_pixel.shape + (3,), np.nan, dtype=float)
    for index, breath in np.ndenumerate(per_pixel):
        if breath is not None:
            landmarks[index] = (breath.start_time, breath.middle_time, breath.end_time)
    files["pixel"] = {
        "tiv/time": np.stack([np.asarray(t, dtype=float) for t in pixel_tiv.time]),
        "tiv/values": np.asarray(pixel_tiv.values, dtype=float),
        "breaths/landmarks": landmarks,
        "breaths/start_end": np.asarray(pixel_breaths.intervals, dtype=float).reshape(
            -1, 2
        ),
    }

    files["roi"] = {}
    notes["captures_left_out"]["roi"] = {}
    methods = (
        (
            "tiv_lungspace",
            TIVLungspace(threshold=SETTINGS["tiv_lungspace"]["threshold"]),
        ),
        (
            "amplitude_lungspace",
            AmplitudeLungspace(threshold=SETTINGS["amplitude_lungspace"]["threshold"]),
        ),
        (
            "watershed_lungspace",
            WatershedLungspace(
                threshold_fraction=SETTINGS["watershed_lungspace"]["threshold_fraction"]
            ),
        ),
    )
    masks = {}
    for name, method in methods:
        captures: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masks[name] = method.apply(
                filtered, timing_data=global_impedance, captures=captures
            )
        record(name, caught)
        files["roi"][f"{name}/mask"] = np.asarray(masks[name].mask, dtype=float)
        notes["captures_left_out"]["roi"][name] = add_captures(
            files["roi"], f"{name}/captures/", captures
        )
    sized = FilterROIBySize(
        min_region_size=SETTINGS["filter_by_size"]["min_region_size"]
    ).apply(masks["watershed_lungspace"])
    files["roi"]["filter_by_size/mask"] = np.asarray(sized.mask, dtype=float)
    return files, notes


def main() -> int:
    """Make the golden files and the manifest.

    Returns:
        int: 0 when all files are written, 1 when another eitprocessing
        version than the reference is installed.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Folder for the golden files."
    )
    args = parser.parse_args()
    output = os.path.abspath(args.output)

    version = importlib.metadata.version("eitprocessing")
    if version != REFERENCE_VERSION:
        print(
            f"eitprocessing {version} is installed; the reference is {REFERENCE_VERSION}."
        )
        return 1

    manifest: dict[str, Any] = {
        "made_on": datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        "eitprocessing": {"version": version, "commit": eitprocessing_commit()},
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-image")
        },
        "settings": SETTINGS,
        "slicing": {"windows_s": SLICE_WINDOWS, "start_end_inclusive": SLICE_FLAGS},
        "comparison": COMPARISON_RULE,
        "inputs": {},
    }
    for file_name, vendor, folder in INPUTS:
        path = os.path.join(DATA_DIR, file_name)
        files, notes = run_chain(path, vendor)
        folder_path = os.path.join(output, folder)
        os.makedirs(folder_path, exist_ok=True)
        written = {}
        for name, file_arrays in files.items():
            arrays: dict[str, Any] = file_arrays
            np.savez_compressed(os.path.join(folder_path, f"{name}.npz"), **arrays)
            written[f"{name}.npz"] = sorted(arrays)
        manifest["inputs"][folder] = {
            "file": os.path.relpath(path, REPO_ROOT).replace(os.sep, "/"),
            "vendor": vendor,
            "sha256": sha256_of_file(path),
            "golden_files": written,
            **notes,
        }
        print(f"{folder}: {len(files)} files written")

    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
