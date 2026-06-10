from __future__ import annotations

import os
import re
import struct
import sys
import types

import numpy as np
import pytest

from m3resp.synthetic import unified_generator as generator


def read_poly5_like_resurfemg(path):
    with open(path, "rb") as file_obj:
        header = struct.unpack(
            "=31sH81phhBHi4xHHHHHHHiHHH64x",
            file_obj.read(217),
        )
        sample_rate = header[3]
        num_channels = header[6] // 2
        num_samples = header[7]
        num_data_blocks = header[15]
        samples_per_block = header[16]

        labels = []
        units = []
        for _ in range(num_channels):
            channel_description = struct.unpack(
                "=41p4x11pffffH62x",
                file_obj.read(136),
            )
            labels.append(channel_description[0][5:].decode("ascii"))
            units.append(channel_description[1].decode("utf-8"))
            file_obj.read(136)

        sample_buffer = np.zeros(num_channels * num_samples, dtype=np.float32)
        offset = 0
        for block_index in range(num_data_blocks):
            remaining_samples = num_samples - block_index * samples_per_block
            block_samples = min(samples_per_block, remaining_samples)
            block_values = block_samples * num_channels
            file_obj.read(86)
            block = np.frombuffer(
                file_obj.read(block_values * 4),
                dtype="<f4",
            )
            sample_buffer[offset : offset + block_values] = block
            offset += block_values

    samples = sample_buffer.reshape(num_samples, num_channels).T
    return {
        "sample_rate": sample_rate,
        "labels": labels,
        "units": units,
        "samples": samples,
    }


def test_eit_only_generation_writes_draeger_and_portable_exports(tmp_path):
    config = generator.SyntheticGeneratorConfig(
        duration_seconds=2.0,
        output_dir=str(tmp_path),
        basename="eit_case",
        timestamp_output_dir=False,
        generate_eit=True,
        generate_emg=False,
        generate_ventilator=False,
        write_native_outputs=False,
        eit=generator.EITGeneratorConfig(sample_frequency_hz=20.0),
    )

    dataset = generator.generate_synthetic_dataset(config)

    assert dataset.eit is not None
    assert dataset.emg is None
    assert dataset.ventilator is None
    assert dataset.eit.array.shape == (40, 32, 32)
    assert dataset.eit.sample_frequency == 20.0
    assert dataset.eit.units == ["a.u."]
    assert dataset.eit.metadata["frame_size_bytes"] == 4358
    assert os.path.exists(dataset.eit.paths["npy"])
    assert os.path.exists(dataset.eit.paths["csv"])
    assert os.path.exists(dataset.eit.paths["components_npz"])
    assert os.path.exists(dataset.eit.paths["native"])
    assert "drift" in dataset.eit.metadata["component_labels"]
    assert os.path.getsize(dataset.eit.paths["native"]) == 40 * 4358
    assert os.path.exists(os.path.join(str(tmp_path), "eit_case_metadata.json"))


def test_drift_disabled_and_enabled_are_deterministic():
    time = np.arange(0, 10, 0.5)

    disabled = generator.generate_drift(time, generator.DriftConfig(enabled=False))
    enabled = generator.generate_drift(
        time,
        generator.DriftConfig(enabled=True, amplitude=0.5, kind="sinusoidal"),
    )
    enabled_again = generator.generate_drift(
        time,
        generator.DriftConfig(enabled=True, amplitude=0.5, kind="sinusoidal"),
    )

    assert np.allclose(disabled, 0.0)
    assert not np.allclose(enabled, 0.0)
    assert np.allclose(enabled, enabled_again)


def test_timing_drift_shifts_arrays_along_selected_sample_axis():
    time = np.arange(0, 5, 1.0)
    values = np.asarray(
        [
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    config = generator.TimingDriftConfig(
        enabled=True,
        time_shift_seconds=2.0,
    )

    shifted_time, shifted = generator.shift_array_in_time(
        values,
        time,
        config,
        sample_axis=1,
    )

    assert np.allclose(shifted_time, np.asarray([2.0, 3.0, 4.0]))
    assert np.allclose(
        shifted,
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float32,
        ),
    )


def test_eit_drift_lives_under_eit_config():
    drift = generator.DriftConfig(
        enabled=True,
        kind="constant",
        amplitude=0.25,
    )
    config = generator.SyntheticGeneratorConfig(
        duration_seconds=2.0,
        generate_emg=False,
        generate_ventilator=False,
        eit=generator.EITGeneratorConfig(
            sample_frequency_hz=2.0,
            noise_std_au=0.0,
            drift=drift,
        ),
    )

    _, _, components = generator.generate_realistic_eit_signal(
        config,
        duration=config.duration_seconds,
        fs=config.eit.sample_frequency_hz,
        seed=config.seed,
    )

    assert np.allclose(components["drift"], 0.25)


def test_emg_and_ventilator_generation_calls_resurfemg_with_config(
    monkeypatch,
    tmp_path,
):
    calls = {}

    def simulate_raw_emg(**kwargs):
        calls.setdefault("emg", []).append(kwargs)
        return np.full(int(kwargs["fs_emg"] * kwargs["t_end"]), kwargs["emg_amp"])

    def simulate_ventilator_data(**kwargs):
        calls["ventilator"] = kwargs
        n_samples = int(kwargs["fs_vent"] * kwargs["t_end"])
        return np.vstack(
            [
                np.full(n_samples, kwargs["dp"]),
                np.arange(n_samples, dtype=float),
                np.zeros(n_samples, dtype=float),
            ]
        ), np.full(n_samples, kwargs["p_mus_amp"])

    synthetic_data = types.ModuleType("resurfemg.pipelines.synthetic_data")
    synthetic_data.simulate_raw_emg = simulate_raw_emg
    synthetic_data.simulate_ventilator_data = simulate_ventilator_data
    pipelines = types.ModuleType("resurfemg.pipelines")
    resurfemg = types.ModuleType("resurfemg")
    monkeypatch.setitem(sys.modules, "resurfemg", resurfemg)
    monkeypatch.setitem(sys.modules, "resurfemg.pipelines", pipelines)
    monkeypatch.setitem(
        sys.modules,
        "resurfemg.pipelines.synthetic_data",
        synthetic_data,
    )

    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        basename="emg_vent",
        timestamp_output_dir=False,
        generate_eit=False,
        generate_emg=True,
        generate_ventilator=True,
        write_native_outputs=False,
        respiratory=generator.RespiratoryPatternConfig(
            respiratory_rate_bpm=22.0,
            ie_ratio=0.5,
            occlusion_times_seconds=(0.25, 0.75),
        ),
        emg=generator.EMGGeneratorConfig(
            sample_frequency_hz=10.0,
            channel_amplitudes_uv=(0.2, 5.0),
            drift_amplitude_uv=7.0,
            noise_amplitude_uv=0.1,
            heart_rate_bpm=80.0,
        ),
        ventilator=generator.VentilatorGeneratorConfig(
            sample_frequency_hz=5.0,
            driving_pressure_cm_h2o=12.0,
            muscle_pressure_amplitude_cm_h2o=3.0,
        ),
    )

    dataset = generator.generate_synthetic_dataset(config)

    assert dataset.emg is not None
    assert dataset.ventilator is not None
    assert dataset.emg.array.shape == (2, 10)
    assert dataset.ventilator.array.shape == (3, 5)
    assert len(calls["emg"]) == 2
    assert calls["emg"][0]["fs_emg"] == 10.0
    assert calls["emg"][0]["rr"] == 22.0
    assert calls["emg"][0]["ie_ratio"] == 0.5
    assert calls["emg"][0]["drift_amp"] == 7.0
    assert calls["emg"][0]["noise_amp"] == 0.1
    assert np.allclose(calls["emg"][0]["t_p_occs"], np.asarray([0.25, 0.75]))
    assert calls["ventilator"]["fs_vent"] == 5.0
    assert calls["ventilator"]["rr"] == 22.0
    assert calls["ventilator"]["dp"] == 12.0
    assert calls["ventilator"]["p_mus_amp"] == 3.0
    assert os.path.exists(dataset.emg.paths["npy"])
    assert os.path.exists(dataset.emg.paths["csv"])
    assert "native" not in dataset.emg.paths
    assert "native" not in dataset.ventilator.paths


def test_emg_timing_drift_does_not_shift_ventilator(monkeypatch, tmp_path):
    def simulate_raw_emg(**kwargs):
        signal = np.arange(int(kwargs["fs_emg"] * kwargs["t_end"]), dtype=float)
        signal[:2] = 0.0
        return signal

    def simulate_ventilator_data(**kwargs):
        n_samples = int(kwargs["fs_vent"] * kwargs["t_end"])
        return np.vstack(
            [
                np.arange(n_samples, dtype=float),
                np.arange(n_samples, dtype=float) + 10.0,
                np.arange(n_samples, dtype=float) + 20.0,
            ]
        ), np.arange(n_samples, dtype=float) + 30.0

    synthetic_data = types.ModuleType("resurfemg.pipelines.synthetic_data")
    synthetic_data.simulate_raw_emg = simulate_raw_emg
    synthetic_data.simulate_ventilator_data = simulate_ventilator_data
    pipelines = types.ModuleType("resurfemg.pipelines")
    resurfemg = types.ModuleType("resurfemg")
    monkeypatch.setitem(sys.modules, "resurfemg", resurfemg)
    monkeypatch.setitem(sys.modules, "resurfemg.pipelines", pipelines)
    monkeypatch.setitem(
        sys.modules,
        "resurfemg.pipelines.synthetic_data",
        synthetic_data,
    )

    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        basename="emg_shift_only",
        timestamp_output_dir=False,
        generate_eit=False,
        generate_emg=True,
        generate_ventilator=True,
        write_native_outputs=False,
        emg=generator.EMGGeneratorConfig(
            sample_frequency_hz=10.0,
            channel_amplitudes_uv=(1.0,),
            timing_drift=generator.TimingDriftConfig(
                enabled=True,
                time_shift_seconds=0.2,
            ),
        ),
        ventilator=generator.VentilatorGeneratorConfig(
            sample_frequency_hz=5.0,
            timing_drift=generator.TimingDriftConfig(enabled=False),
        ),
    )

    dataset = generator.generate_synthetic_dataset(config)

    assert dataset.emg is not None
    assert dataset.ventilator is not None
    assert np.allclose(dataset.emg.time, np.arange(0.2, 1.0, 0.1))
    assert np.allclose(
        dataset.emg.array[0],
        np.asarray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )
    assert np.allclose(dataset.ventilator.time, np.arange(5, dtype=float) / 5.0)
    assert np.allclose(dataset.ventilator.array[0], np.arange(5, dtype=float))


def test_resurfemg_length_mismatch_retries_and_normalizes_signal(
    monkeypatch,
    tmp_path,
):
    calls = []

    def simulate_raw_emg(**kwargs):
        calls.append(kwargs)
        if kwargs["ecg_acceleration"] != 1.0:
            raise ValueError(
                "operands could not be broadcast together with shapes (10,) (8,)"
            )
        return np.arange(8, dtype=float)

    synthetic_data = types.ModuleType("resurfemg.pipelines.synthetic_data")
    synthetic_data.simulate_raw_emg = simulate_raw_emg
    pipelines = types.ModuleType("resurfemg.pipelines")
    resurfemg = types.ModuleType("resurfemg")
    monkeypatch.setitem(sys.modules, "resurfemg", resurfemg)
    monkeypatch.setitem(sys.modules, "resurfemg.pipelines", pipelines)
    monkeypatch.setitem(
        sys.modules,
        "resurfemg.pipelines.synthetic_data",
        synthetic_data,
    )

    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        basename="retry_case",
        timestamp_output_dir=False,
        generate_eit=False,
        generate_emg=True,
        generate_ventilator=False,
        write_native_outputs=False,
        emg=generator.EMGGeneratorConfig(
            sample_frequency_hz=10.0,
            channel_amplitudes_uv=(5.0,),
            ecg_acceleration=1.6,
        ),
    )

    dataset = generator.generate_synthetic_dataset(config)

    assert len(calls) == 2
    assert calls[0]["ecg_acceleration"] == 1.6
    assert calls[1]["ecg_acceleration"] == 1.0
    assert dataset.emg.array.shape == (1, 10)
    assert dataset.emg.array[0, -1] == 7


def test_yaml_config_loader_builds_typed_config(tmp_path):
    config_path = os.path.join(str(tmp_path), "synthetic.yaml")
    output_dir = os.path.join("relative", "output")
    with open(config_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(
            "\n".join(
                [
                    "duration_seconds: 3.0",
                    "seed: 7",
                    f"output_dir: {output_dir}",
                    "basename: yaml_case",
                    "timestamp_output_dir: false",
                    "generate_eit: false",
                    "generate_emg: true",
                    "generate_ventilator: false",
                    "write_native_outputs: false",
                    "respiratory:",
                    "  respiratory_rate_bpm: 18.0",
                    "  occlusion_times_seconds:",
                    "    - 1.0",
                    "    - 2.0",
                    "eit:",
                    "  drift:",
                    "    enabled: true",
                    "    kind: linear",
                    "    amplitude: 0.2",
                    "emg:",
                    "  sample_frequency_hz: 1000",
                    "  channel_amplitudes_uv:",
                    "    - 4.0",
                    "  timing_drift:",
                    "    enabled: true",
                    "    time_shift_seconds: -0.5",
                ]
            )
        )

    config = generator.load_synthetic_generator_config(config_path)

    assert config.duration_seconds == 3.0
    assert config.seed == 7
    assert config.basename == "yaml_case"
    assert config.timestamp_output_dir is False
    assert config.output_dir == os.path.normpath(
        os.path.join(str(tmp_path), output_dir)
    )
    assert config.respiratory.occlusion_times_seconds == (1.0, 2.0)
    assert config.eit.drift.enabled is True
    assert config.eit.drift.kind == "linear"
    assert config.eit.drift.amplitude == 0.2
    assert config.emg.channel_amplitudes_uv == (4.0,)
    assert config.emg.timing_drift is not None
    assert config.emg.timing_drift.enabled is True
    assert config.emg.timing_drift.time_shift_seconds == -0.5


def test_yaml_config_loader_rejects_top_level_drift():
    with pytest.raises(ValueError, match="Unknown synthetic generator config keys"):
        generator.synthetic_generator_config_from_dict({"drift": {"enabled": True}})


def test_main_loads_yaml_path(monkeypatch, tmp_path, capsys):
    config_path = os.path.join(str(tmp_path), "custom.yaml")
    loaded_paths = []

    def fake_load(path):
        loaded_paths.append(path)
        return generator.SyntheticGeneratorConfig(
            output_dir=str(tmp_path),
            basename="main_case",
            timestamp_output_dir=False,
            generate_eit=False,
            generate_emg=False,
            generate_ventilator=False,
            write_native_outputs=False,
        )

    def fake_generate(config):
        return generator.SyntheticDataset(
            provenance={
                "output_root": config.output_dir,
                "output_dir": config.output_dir,
            }
        )

    monkeypatch.setattr(generator, "load_synthetic_generator_config", fake_load)
    monkeypatch.setattr(generator, "generate_synthetic_dataset", fake_generate)

    generator.main(config_path)
    output = capsys.readouterr().out

    assert loaded_paths == [config_path]
    assert "custom.yaml" in output


def test_timestamp_output_directory_creates_run_folder(tmp_path):
    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        basename="timestamp_case",
        timestamp_output_dir=True,
        generate_eit=False,
        generate_emg=False,
        generate_ventilator=False,
        write_native_outputs=False,
    )

    dataset = generator.generate_synthetic_dataset(config)

    output_dir = dataset.provenance["output_dir"]
    assert os.path.dirname(output_dir) == str(tmp_path)
    assert re.match(r"\d{8}_\d{6}(?:_\d{2})?$", os.path.basename(output_dir))
    assert os.path.exists(os.path.join(output_dir, "timestamp_case_metadata.json"))


def test_missing_resurfemg_raises_clear_error(monkeypatch, tmp_path):
    original_import_module = generator.importlib.import_module

    def raise_import_error(module_name):
        if module_name == "resurfemg.pipelines.synthetic_data":
            raise ImportError("missing resurfemg")
        return original_import_module(module_name)

    monkeypatch.setattr(generator.importlib, "import_module", raise_import_error)

    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        timestamp_output_dir=False,
        generate_eit=False,
        generate_emg=True,
        generate_ventilator=False,
    )

    with pytest.raises(RuntimeError, match="requires the optional dependency"):
        generator.generate_synthetic_dataset(config)


def test_native_emg_and_ventilator_request_writes_poly5_without_upstream_writer(
    monkeypatch,
    tmp_path,
):

    def simulate_raw_emg(**kwargs):
        return np.arange(int(kwargs["fs_emg"] * kwargs["t_end"]), dtype=float)

    def simulate_ventilator_data(**kwargs):
        n_samples = int(kwargs["fs_vent"] * kwargs["t_end"])
        return np.vstack(
            [
                np.arange(n_samples, dtype=float),
                np.arange(n_samples, dtype=float) + 10,
                np.arange(n_samples, dtype=float) + 20,
            ]
        ), np.full(n_samples, kwargs["p_mus_amp"])

    synthetic_data = types.ModuleType("resurfemg.pipelines.synthetic_data")
    synthetic_data.simulate_raw_emg = simulate_raw_emg
    synthetic_data.simulate_ventilator_data = simulate_ventilator_data
    pipelines = types.ModuleType("resurfemg.pipelines")
    resurfemg = types.ModuleType("resurfemg")
    monkeypatch.setitem(sys.modules, "resurfemg", resurfemg)
    monkeypatch.setitem(sys.modules, "resurfemg.pipelines", pipelines)
    monkeypatch.setitem(
        sys.modules,
        "resurfemg.pipelines.synthetic_data",
        synthetic_data,
    )

    config = generator.SyntheticGeneratorConfig(
        duration_seconds=1.0,
        output_dir=str(tmp_path),
        basename="native_missing",
        timestamp_output_dir=False,
        generate_eit=False,
        generate_emg=True,
        generate_ventilator=True,
        write_native_outputs=True,
        emg=generator.EMGGeneratorConfig(
            sample_frequency_hz=10.0,
            channel_amplitudes_uv=(1.0, 2.0),
        ),
        ventilator=generator.VentilatorGeneratorConfig(sample_frequency_hz=5.0),
    )

    dataset = generator.generate_synthetic_dataset(config)

    assert dataset.emg is not None
    assert dataset.ventilator is not None

    assert os.path.exists(os.path.join(str(tmp_path), "native_missing_emg.npy"))
    assert os.path.exists(os.path.join(str(tmp_path), "native_missing_emg.csv"))
    assert os.path.exists(dataset.emg.paths["native"])
    assert os.path.exists(dataset.ventilator.paths["native"])

    emg_poly5 = read_poly5_like_resurfemg(dataset.emg.paths["native"])
    assert emg_poly5["sample_rate"] == 10
    assert emg_poly5["labels"] == ["emg_0", "emg_1"]
    assert emg_poly5["units"] == ["uV", "uV"]
    assert np.allclose(emg_poly5["samples"], dataset.emg.array)

    vent_poly5 = read_poly5_like_resurfemg(dataset.ventilator.paths["native"])
    assert vent_poly5["sample_rate"] == 5
    assert vent_poly5["labels"] == ["pressure", "flow", "volume"]
    assert vent_poly5["units"] == ["cmH2O", "L/s", "L"]
    assert np.allclose(vent_poly5["samples"], dataset.ventilator.array)
