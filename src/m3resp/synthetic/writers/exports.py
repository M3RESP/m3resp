"""Generic record/JSON/CSV export helpers and run-output directory resolution."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from m3resp.synthetic.config import ExportConfig
from m3resp.synthetic.writers.poly5 import _write_poly5_file

DEFAULT_ONE = 1.0
DEFAULT_ONE_DIMENSION = 1
DEFAULT_FIRST_AXIS = 0
DEFAULT_SECOND_AXIS = 1
DEFAULT_ROW_VECTOR_SHAPE = (1, -1)
DEFAULT_METADATA_INDENT = 2


def _resolve_run_output_dir(
    output_root: str,
    timestamp_output_dir: bool,
    export_config: ExportConfig,
) -> str:
    if not timestamp_output_dir:
        return output_root

    timestamp = datetime.now().strftime(export_config.timestamp_format)
    candidate = os.path.join(output_root, timestamp)
    suffix = int(DEFAULT_ONE)
    while os.path.exists(candidate):
        formatted_suffix = f"{suffix:0{export_config.timestamp_suffix_width}d}"
        candidate = os.path.join(output_root, f"{timestamp}_{formatted_suffix}")
        suffix += int(DEFAULT_ONE)
    return candidate


def _write_record_exports(
    output_dir: str,
    basename: str,
    time: np.ndarray,
    array: np.ndarray,
    labels: list[str],
    export_config: ExportConfig,
) -> dict[str, str]:
    npy_path = os.path.join(output_dir, f"{basename}.npy")
    csv_path = os.path.join(output_dir, f"{basename}.csv")
    np.save(npy_path, array)
    _write_timeseries_csv(csv_path, time, array, labels, export_config)
    return {"npy": npy_path, "csv": csv_path}


def _write_timeseries_csv(
    path: str,
    time: np.ndarray,
    array: np.ndarray,
    labels: list[str],
    export_config: ExportConfig,
) -> None:
    values = np.asarray(array)
    if values.ndim == DEFAULT_ONE_DIMENSION:
        values = values.reshape(DEFAULT_ROW_VECTOR_SHAPE)
    if values.shape[DEFAULT_FIRST_AXIS] != len(labels):
        raise ValueError("labels length must match the first data dimension")
    if values.shape[DEFAULT_SECOND_AXIS] != len(time):
        raise ValueError("time length must match the second data dimension")

    with open(path, "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([export_config.csv_time_label, *labels])
        for index, time_value in enumerate(time):
            row_values = [float(value) for value in values[:, index]]
            writer.writerow([float(time_value), *row_values])


def _write_native_resurfemg_output(
    synth: Any,
    output_dir: str,
    basename: str,
    kind: str,
    time: np.ndarray,
    array: np.ndarray,
    sample_frequency: float,
    labels: list[str],
    units: list[str],
    export_config: ExportConfig,
) -> str | None:
    writer = getattr(synth, "write_synthetic_recording", None)

    native_path = os.path.join(
        output_dir,
        f"{basename}{export_config.native_extension}",
    )
    if writer is None:
        _write_poly5_file(
            path=native_path,
            array=array,
            sample_frequency=sample_frequency,
            labels=labels,
            units=units,
        )
        return native_path

    result = writer(
        path=native_path,
        kind=kind,
        time=time,
        array=array,
        sample_frequency=sample_frequency,
        labels=labels,
        units=units,
    )
    if result is not None:
        native_path = str(result)
    return native_path


def _write_json(
    path: str,
    data: dict[str, Any],
    export_config: ExportConfig | None = None,
) -> None:
    indent = (
        DEFAULT_METADATA_INDENT
        if export_config is None
        else export_config.metadata_indent
    )
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=indent, sort_keys=True)
