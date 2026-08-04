"""Top-level orchestration for synthetic EIT/EMG/ventilator dataset generation.

This module provides a single typed configuration surface for generating
reproducible synthetic respiratory datasets.
"""

from __future__ import annotations

import importlib  # noqa: F401 - re-exported so callers can patch import_module
import json
import os
import sys
from typing import Any, cast

import numpy as np

from m3resp.synthetic.config import (
    DriftConfig,
    EITGeneratorConfig,
    EMGGeneratorConfig,
    RespiratoryPatternConfig,
    SyntheticDataset,
    SyntheticDraegerData,
    SyntheticGeneratorConfig,
    SyntheticRecord,
    TimingDriftConfig,
    VentilatorGeneratorConfig,
    _config_to_dict,
    _dataset_metadata,
    load_synthetic_generator_config,
    synthetic_generator_config_from_dict,
)
from m3resp.synthetic.drift import generate_drift, shift_array_in_time
from m3resp.synthetic.vendor import (
    _load_resurfemg_synthetic,
    _seeded_vendor_randomness,
    _simulate_raw_emg_length_safe,
)
from m3resp.synthetic.waveforms import (
    detect_min_max_flags,
    generate_event_series,
    generate_medibus_data,
    generate_realistic_eit_signal,
    signal_to_pixel_impedance,
)
from m3resp.synthetic.writers.draeger import pack_draeger_frame, save_draeger_bin
from m3resp.synthetic.writers.exports import (
    _resolve_run_output_dir,
    _write_json,
    _write_native_resurfemg_output,
    _write_record_exports,
    _write_timeseries_csv,
)

__all__ = [
    "FORMAT_SPECS",
    "DriftConfig",
    "EITGeneratorConfig",
    "EMGGeneratorConfig",
    "RespiratoryPatternConfig",
    "TimingDriftConfig",
    "VentilatorGeneratorConfig",
    "generate_drift",
    "generate_eit_record",
    "generate_emg_record",
    "generate_synthetic_dataset",
    "generate_synthetic_draeger_data",
    "generate_ventilator_record",
    "main",
    "pack_draeger_frame",
    "save_draeger_bin",
    "synthetic_generator_config_from_dict",
]

FORMAT_SPECS = {
    "original": {
        "frame_size": 4358,
        "n_medibus_fields": 52,
    },
    "pressure_pod": {
        "frame_size": 4382,
        "n_medibus_fields": 58,
    },
}
DEFAULT_GENERATOR_MODULE = "m3resp.synthetic.generator"
DEFAULT_CONFIG_FILENAME = "synthetic_generator_config.yaml"
DEFAULT_EIT_LABELS = ("pixel_impedance",)
DEFAULT_EIT_UNITS = ("a.u.",)
DEFAULT_EMG_UNIT = "uV"
DEFAULT_VENTILATOR_LABELS = ("pressure", "flow", "volume")
DEFAULT_VENTILATOR_UNITS = ("cmH2O", "L/s", "L")
DEFAULT_RANDOM_HEART_RATE_MIN_BPM = 60
DEFAULT_RANDOM_HEART_RATE_MAX_BPM = 100
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_FLOAT64_DTYPE = np.float64
DEFAULT_INT32_DTYPE = np.int32
DEFAULT_GLOBAL_IMPEDANCE_AXES = (1, 2)
DEFAULT_GLOBAL_IMPEDANCE_LABEL = "global_impedance"
DEFAULT_ROW_VECTOR_SHAPE = (1, -1)
DEFAULT_ONE_DIMENSION = 1
DEFAULT_FIRST_AXIS = 0
DEFAULT_SECOND_AXIS = 1


def generate_synthetic_dataset(config: SyntheticGeneratorConfig) -> SyntheticDataset:
    """Generate synthetic EIT, EMG, and ventilator data from one configuration."""

    output_root = os.path.abspath(os.path.normpath(config.output_dir))
    output_dir = _resolve_run_output_dir(
        output_root=output_root,
        timestamp_output_dir=config.timestamp_output_dir,
        export_config=config.export,
    )
    os.makedirs(output_dir, exist_ok=True)

    provenance = {
        "generator": DEFAULT_GENERATOR_MODULE,
        "config": _config_to_dict(config),
        "output_root": output_root,
        "output_dir": output_dir,
    }
    dataset = SyntheticDataset(provenance=provenance)

    if config.generate_eit:
        dataset.eit = generate_eit_record(config, output_dir)

    if config.generate_emg:
        dataset.emg = generate_emg_record(config, output_dir)

    if config.generate_ventilator:
        dataset.ventilator = generate_ventilator_record(config, output_dir)

    metadata_path = os.path.join(output_dir, f"{config.basename}_metadata.json")
    _write_json(metadata_path, _dataset_metadata(dataset), config.export)
    provenance["metadata_path"] = metadata_path
    return dataset


def generate_eit_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate EIT waveform, 32x32 frames, and portable plus Draeger outputs."""

    draeger = generate_synthetic_draeger_data(
        config=config,
        duration=config.duration_seconds,
        fs=config.eit.sample_frequency_hz,
        seed=config.seed,
    )
    global_impedance = draeger.pixel_impedance.sum(
        axis=DEFAULT_GLOBAL_IMPEDANCE_AXES
    ).astype(DEFAULT_FLOAT32_DTYPE)

    npy_path = os.path.join(output_dir, f"{config.basename}_eit_pixels.npy")
    csv_path = os.path.join(output_dir, f"{config.basename}_eit_global.csv")
    bin_path = os.path.join(output_dir, f"{config.basename}_eit_draeger.bin")
    components_path = os.path.join(
        output_dir,
        f"{config.basename}_eit_components.npz",
    )

    np.save(npy_path, draeger.pixel_impedance)
    np.savez(components_path, **cast(dict[str, Any], draeger.components))
    _write_timeseries_csv(
        csv_path,
        draeger.time,
        global_impedance.reshape(DEFAULT_ROW_VECTOR_SHAPE),
        [DEFAULT_GLOBAL_IMPEDANCE_LABEL],
        config.export,
    )
    save_draeger_bin(bin_path, draeger, config.export)

    return SyntheticRecord(
        time=draeger.time,
        array=draeger.pixel_impedance,
        sample_frequency=draeger.sample_frequency,
        labels=list(DEFAULT_EIT_LABELS),
        units=list(DEFAULT_EIT_UNITS),
        paths={
            "npy": npy_path,
            "csv": csv_path,
            "components_npz": components_path,
            "native": bin_path,
        },
        metadata={
            "modality": "eit",
            "vendor": "draeger",
            "format_name": draeger.format_name,
            "frame_size_bytes": draeger.frame_size,
            "time_units": "seconds",
            "signal_units": "relative impedance (a.u.)",
            "component_labels": sorted(draeger.components),
        },
    )


def generate_emg_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate EMG data with ReSurfEMG and write portable exports."""

    synth = _load_resurfemg_synthetic()
    rng = np.random.default_rng(config.seed)
    fs_emg = round(config.emg.sample_frequency_hz)
    expected_samples = round(config.duration_seconds * fs_emg)
    channels = []
    for channel_index, amplitude_uv in enumerate(config.emg.channel_amplitudes_uv):
        heart_rate_bpm = round(config.emg.heart_rate_bpm)
        if config.emg.heart_rate_bpm <= 0:
            heart_rate_bpm = int(
                rng.integers(
                    DEFAULT_RANDOM_HEART_RATE_MIN_BPM,
                    DEFAULT_RANDOM_HEART_RATE_MAX_BPM + 1,
                )
            )
        with _seeded_vendor_randomness(config.seed + channel_index):
            signal = _simulate_raw_emg_length_safe(
                synth=synth,
                expected_samples=expected_samples,
                t_p_occs=np.asarray(
                    config.respiratory.occlusion_times_seconds,
                    dtype=float,
                ),
                t_end=config.duration_seconds,
                fs_emg=fs_emg,
                rr=config.respiratory.respiratory_rate_bpm,
                ie_ratio=config.respiratory.ie_ratio,
                tau_mus_up=config.emg.tau_mus_up_seconds,
                tau_mus_down=config.emg.tau_mus_down_seconds,
                emg_amp=amplitude_uv,
                drift_amp=config.emg.drift_amplitude_uv,
                noise_amp=config.emg.noise_amplitude_uv,
                heart_rate=heart_rate_bpm,
                ecg_acceleration=config.emg.ecg_acceleration,
            )
        channels.append(signal)

    array = np.asarray(channels, dtype=np.float32)
    time = np.arange(array.shape[1], dtype=float) / fs_emg
    time, array = shift_array_in_time(
        array,
        time,
        config.emg.timing_drift,
        sample_axis=DEFAULT_SECOND_AXIS,
    )
    labels = [f"emg_{index}" for index in range(array.shape[0])]
    paths = _write_record_exports(
        output_dir=output_dir,
        basename=f"{config.basename}_emg",
        time=time,
        array=array,
        labels=labels,
        export_config=config.export,
    )

    if config.write_native_outputs:
        native_path = _write_native_resurfemg_output(
            synth=synth,
            output_dir=output_dir,
            basename=f"{config.basename}_emg",
            kind="emg",
            time=time,
            array=array,
            sample_frequency=fs_emg,
            labels=labels,
            units=[DEFAULT_EMG_UNIT] * len(labels),
            export_config=config.export,
        )
        if native_path is not None:
            paths["native"] = native_path

    return SyntheticRecord(
        time=time,
        array=array,
        sample_frequency=fs_emg,
        labels=labels,
        units=[DEFAULT_EMG_UNIT] * len(labels),
        paths=paths,
        metadata={
            "modality": "emg",
            "time_units": "seconds",
            "signal_units": DEFAULT_EMG_UNIT,
        },
    )


def generate_ventilator_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate ventilator data with ReSurfEMG and write portable exports."""

    synth = _load_resurfemg_synthetic()
    fs_vent = round(config.ventilator.sample_frequency_hz)
    with _seeded_vendor_randomness(config.seed):
        y_vent, p_mus = synth.simulate_ventilator_data(
            t_end=config.duration_seconds,
            fs_vent=fs_vent,
            p_mus_amp=config.ventilator.muscle_pressure_amplitude_cm_h2o,
            rr=config.respiratory.respiratory_rate_bpm,
            dp=config.ventilator.driving_pressure_cm_h2o,
            t_p_occs=np.asarray(
                config.respiratory.occlusion_times_seconds, dtype=float
            ),
        )

    array = np.asarray(y_vent, dtype=np.float32)
    p_mus = np.asarray(p_mus, dtype=np.float32)
    time = np.arange(array.shape[1], dtype=float) / fs_vent
    original_time = time
    time, array = shift_array_in_time(
        array,
        original_time,
        config.ventilator.timing_drift,
        sample_axis=DEFAULT_SECOND_AXIS,
    )
    _, p_mus = shift_array_in_time(
        p_mus,
        original_time,
        config.ventilator.timing_drift,
        sample_axis=DEFAULT_FIRST_AXIS,
    )
    labels = list(DEFAULT_VENTILATOR_LABELS)
    paths = _write_record_exports(
        output_dir=output_dir,
        basename=f"{config.basename}_ventilator",
        time=time,
        array=array,
        labels=labels,
        export_config=config.export,
    )
    p_mus_path = os.path.join(output_dir, f"{config.basename}_ventilator_p_mus.npy")
    np.save(p_mus_path, p_mus)
    paths["p_mus_npy"] = p_mus_path

    if config.write_native_outputs:
        native_path = _write_native_resurfemg_output(
            synth=synth,
            output_dir=output_dir,
            basename=f"{config.basename}_ventilator",
            kind="ventilator",
            time=time,
            array=array,
            sample_frequency=fs_vent,
            labels=labels,
            units=list(DEFAULT_VENTILATOR_UNITS),
            export_config=config.export,
        )
        if native_path is not None:
            paths["native"] = native_path

    return SyntheticRecord(
        time=time,
        array=array,
        sample_frequency=fs_vent,
        labels=labels,
        units=list(DEFAULT_VENTILATOR_UNITS),
        paths=paths,
        metadata={
            "modality": "ventilator",
            "time_units": "seconds",
            "p_mus_npy": p_mus_path,
        },
    )


def generate_synthetic_draeger_data(
    config: SyntheticGeneratorConfig,
    duration: float,
    fs: float,
    seed: int,
) -> SyntheticDraegerData:
    """Generate all arrays needed for a parser-compatible Draeger file."""

    if config.eit.format_name not in FORMAT_SPECS:
        raise ValueError(f"Unsupported format_name: {config.eit.format_name}")

    spec = FORMAT_SPECS[config.eit.format_name]
    time_seconds, signal, components = generate_realistic_eit_signal(
        config=config,
        duration=duration,
        fs=fs,
        seed=seed,
    )
    pixel_impedance = signal_to_pixel_impedance(
        signal,
        seed=seed + 1,
        waveform_config=config.eit_waveform,
        template_config=config.lung_template,
    )
    medibus_data = generate_medibus_data(
        time_seconds,
        signal,
        n_medibus_fields=spec["n_medibus_fields"],
        respiratory_rate_bpm=config.respiratory.respiratory_rate_bpm,
        config=config.medibus,
    )
    min_max_flags = detect_min_max_flags(signal)
    event_markers, event_texts = generate_event_series(time_seconds, config.events)

    data = SyntheticDraegerData(
        format_name=config.eit.format_name,
        frame_size=spec["frame_size"],
        sample_frequency=fs,
        time=time_seconds.astype(DEFAULT_FLOAT64_DTYPE),
        pixel_impedance=pixel_impedance.astype(DEFAULT_FLOAT32_DTYPE),
        medibus_data=medibus_data.astype(DEFAULT_FLOAT32_DTYPE),
        min_max_flags=min_max_flags,
        event_markers=event_markers,
        event_texts=event_texts,
        timing_errors=np.zeros(len(time_seconds), dtype=DEFAULT_INT32_DTYPE),
        unused_float32=np.zeros(len(time_seconds), dtype=DEFAULT_FLOAT32_DTYPE),
        components=components,
    )
    return data


def main(config_path: str | None = None) -> None:
    """Generate a synthetic dataset from YAML configuration."""

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            DEFAULT_CONFIG_FILENAME,
        )

    config = load_synthetic_generator_config(config_path)
    dataset = generate_synthetic_dataset(config)
    print(
        json.dumps(
            {
                "config_path": os.path.abspath(os.path.normpath(config_path)),
                **_dataset_metadata(dataset),
            },
            indent=config.export.metadata_indent,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > DEFAULT_ONE_DIMENSION else None)
