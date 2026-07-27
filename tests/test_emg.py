from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from m3resp import BreathEvent, M3Session
from m3resp.adapters import ReSurfEMGAdapter
from m3resp.io import load_emg
from m3resp.modalities.emg import load as load_emg_recording
from m3resp.visualization import (
    plot_session_overview,
    plot_synchronization_comparison,
)


def fake_emg_recording() -> dict[str, Any]:
    return {
        "array": [[0.0, 1.0, 0.0, -1.0]],
        "dataframe": {"kind": "fake-dataframe"},
        "metadata": {"fs": 1000.0, "labels": ["EMG"], "units": ["uV"]},
    }


def test_load_emg_sets_preferred_and_legacy_session_slots():
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(
            loader=lambda *args, **kwargs: fake_emg_recording()
        )
    )

    returned = session.load_emg("subject.Poly5")

    assert returned == fake_emg_recording()
    assert session.emg is session.raw["emg"]
    assert session.emg.raw == fake_emg_recording()["array"]
    assert session.emg.dataframe == {"kind": "fake-dataframe"}
    assert session.emg.metadata["fs"] == 1000.0


def test_top_level_and_modality_load_helpers_return_recordings():
    adapter = ReSurfEMGAdapter(loader=lambda *args, **kwargs: fake_emg_recording())

    top_level = load_emg("subject.Poly5", adapter=adapter)
    modality_level = load_emg_recording("subject.Poly5", adapter=adapter)

    assert top_level.raw == fake_emg_recording()["array"]
    assert modality_level.metadata["labels"] == ["EMG"]


def test_custom_emg_detector_normalization_still_works():
    adapter = ReSurfEMGAdapter()

    events = adapter.detect_breaths(
        {"processed": True},
        detector=lambda data: [(1.0, 2.0, 1.5)],
    )

    assert events == [
        BreathEvent(
            modality="emg",
            start_time=1.0,
            end_time=2.0,
            peak_time=1.5,
            source="resurfemg",
        )
    ]


def test_custom_emg_preprocess_callable_still_works():
    adapter = ReSurfEMGAdapter()

    processed = adapter.preprocess(
        {"recording": True},
        preprocess=lambda signal, *, gain: {"signal": signal, "gain": gain},
        gain=2.0,
    )

    assert processed == {"signal": {"recording": True}, "gain": 2.0}


def test_custom_emg_compute_callable_still_works():
    adapter = ReSurfEMGAdapter()
    events = [BreathEvent("emg", 0.0, 1.0, peak_time=0.5)]

    features = adapter.compute_features(
        {"processed": True},
        events,
        compute=lambda signal, detected_events, *, scale: {
            "signal": signal,
            "events": detected_events,
            "scale": scale,
        },
        scale=3.0,
    )

    assert features == {
        "signal": {"processed": True},
        "events": events,
        "scale": 3.0,
    }


def test_custom_emg_postprocess_callable_still_works():
    adapter = ReSurfEMGAdapter()
    events = [BreathEvent("emg", 0.0, 1.0, peak_time=0.5)]

    result = adapter.postprocess(
        {"processed": True},
        events=events,
        postprocess=lambda processed, *, events, label: {
            "processed": processed,
            "events": events,
            "label": label,
        },
        label="custom",
    )

    assert result == {
        "processed": {"processed": True},
        "events": events,
        "label": "custom",
    }


def test_default_preprocess_updates_emg_recording_with_fake_signal():
    pytest.importorskip("resurfemg")
    np = pytest.importorskip("numpy")

    fs = 1000.0
    time = np.arange(5000, dtype=float) / fs
    fake_signal = np.sin(2 * np.pi * 100 * time)
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(
            loader=lambda *args, **kwargs: {
                "array": np.asarray([fake_signal]),
                "dataframe": {"kind": "fake-dataframe"},
                "metadata": {"fs": fs, "labels": ["EMG"], "units": ["uV"]},
            }
        )
    )

    session.load_emg("subject.Poly5")
    processed = session.preprocess_emg()

    assert len(processed["raw_channel"]) == len(fake_signal)
    assert len(processed["filtered"]) == len(fake_signal)
    assert len(processed["envelope"]) == len(fake_signal)
    assert session.emg.filtered is processed["filtered"]
    assert session.emg.envelope is processed["envelope"]


def test_emg_overview_y_axis_labels_include_amplitude_and_units():
    plt = pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.processed["emg"] = {
        "channel": 0,
        "fs": 1000.0,
        "metadata": {"labels": ["EMG"], "units": ["uV"]},
        "raw_channel": [0.0, 1.0, 0.0],
        "filtered": [0.0, 0.5, 0.0],
        "envelope": [0.0, 0.25, 0.0],
    }

    fig = plot_session_overview(session, max_seconds=None)

    try:
        assert [ax.get_ylabel() for ax in fig.axes] == [
            "EMG amplitude (uV)",
            "EMG amplitude (uV)",
            "EMG amplitude (uV)",
        ]
    finally:
        plt.close(fig)


def test_emg_overview_uses_the_preprocessed_channel_label_and_unit():
    plt = pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.processed["emg"] = {
        "channel": 1,
        "fs": 1000.0,
        "metadata": {
            "labels": ["unused", "diaphragm"],
            "units": ["mV", "uV"],
        },
        "envelope": [0.0, 0.25, 0.0],
    }

    fig = plot_session_overview(session, max_seconds=None)

    try:
        assert fig.axes[0].get_title() == "EMG envelope (diaphragm)"
        assert fig.axes[0].get_ylabel() == "EMG amplitude (uV)"
    finally:
        plt.close(fig)


def test_emg_overview_rejects_a_channel_that_was_not_preprocessed():
    pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.processed["emg"] = {
        "channel": 0,
        "fs": 1000.0,
        "metadata": {"labels": ["diaphragm", "intercostal"], "units": ["uV"]},
        "envelope": [0.0, 0.25, 0.0],
    }

    with pytest.raises(ValueError, match="available processed data is for channel 0"):
        plot_session_overview(session, emg_channel=1)


def test_synchronization_comparison_shifts_signal_time_by_alignment_offset():
    plt = pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.processed["emg"] = {
        "channel": 0,
        "fs": 1000.0,
        "metadata": {"labels": ["EMG"], "units": ["uV"]},
        "envelope": [0.0, 0.0, 1.0, 2.0, 3.0],
    }
    session.add_events(
        "emg_breaths",
        [BreathEvent("emg", 0.002, 0.004, peak_time=0.003)],
    )
    session.align_modalities(offset_seconds={"emg": -0.002})

    fig = plot_synchronization_comparison(session, max_seconds=None)

    try:
        before_ax, after_ax = fig.axes[:2]
        assert list(before_ax.lines[0].get_xdata()) == [
            0.0,
            0.001,
            0.002,
            0.003,
            0.004,
        ]
        assert list(after_ax.lines[0].get_xdata()) == [0.0, 0.001, 0.002]
        assert list(after_ax.lines[0].get_ydata()) == [1.0, 2.0, 3.0]
        assert before_ax.get_title() == "EMG envelope (EMG) before synchronization"
        assert after_ax.get_title() == "EMG envelope (EMG) after synchronization"
    finally:
        plt.close(fig)


def test_synchronization_comparison_uses_raw_eit_signal_when_filtered_exists():
    plt = pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(
        data=SimpleNamespace(
            continuous_data={
                "global_impedance_(raw)": SimpleNamespace(
                    time=[0.0, 1.0],
                    values=[10.0, 11.0],
                    label="raw impedance",
                )
            }
        )
    )
    session.processed["eit"] = {
        "filtered_global_impedance": SimpleNamespace(
            time=[0.0, 1.0],
            values=[100.0, 101.0],
            label="filtered impedance",
        )
    }
    session.add_events(
        "eit_breaths",
        [BreathEvent("eit", 0.0, 1.0, peak_time=0.5)],
    )
    session.align_modalities(offset_seconds={"eit": 0.0})

    fig = plot_synchronization_comparison(session, max_seconds=None)

    try:
        before_ax, after_ax = fig.axes[:2]
        assert before_ax.get_title() == (
            "EIT raw global impedance before synchronization"
        )
        assert after_ax.get_title() == "EIT raw global impedance after synchronization"
        assert list(before_ax.lines[0].get_ydata()) == [10.0, 11.0]
        assert list(after_ax.lines[0].get_ydata()) == [10.0, 11.0]
    finally:
        plt.close(fig)


def test_synchronization_comparison_uses_raw_sync_snapshots_when_available():
    plt = pytest.importorskip("matplotlib.pyplot")
    session = M3Session()
    session.processed["raw_synchronization"] = {
        "emg": {
            "before": {
                "title": "EMG raw (EMG)",
                "time": [0.0, 0.001, 0.002, 0.003],
                "values": [0.0, 0.0, 1.0, 2.0],
                "ylabel": "EMG amplitude (uV)",
            },
            "after": {
                "title": "EMG raw (EMG)",
                "time": [0.002, 0.003],
                "values": [1.0, 2.0],
                "ylabel": "EMG amplitude (uV)",
            },
        },
        "vent_pressure": {
            "before": {
                "title": "Ventilator pressure",
                "time": [0.0, 0.01, 0.02],
                "values": [8.0, 9.0, 10.0],
                "ylabel": "Pressure",
            },
            "after": {
                "title": "Ventilator pressure",
                "time": [0.01, 0.02],
                "values": [9.0, 10.0],
                "ylabel": "Pressure",
            },
        },
        "vent_flow": {
            "before": {
                "title": "Ventilator flow",
                "time": [0.0, 0.01, 0.02],
                "values": [0.0, 0.5, 0.0],
                "ylabel": "Flow",
            },
            "after": {
                "title": "Ventilator flow",
                "time": [0.01, 0.02],
                "values": [0.5, 0.0],
                "ylabel": "Flow",
            },
        },
        "vent_volume": {
            "before": {
                "title": "Ventilator volume",
                "time": [0.0, 0.01, 0.02],
                "values": [100.0, 125.0, 150.0],
                "ylabel": "Volume",
            },
            "after": {
                "title": "Ventilator volume",
                "time": [0.01, 0.02],
                "values": [125.0, 150.0],
                "ylabel": "Volume",
            },
        },
    }
    session.parameters["raw_alignment"] = {
        "offset_seconds": {"eit": 0.0, "emg": -0.002, "vent": 0.0}
    }
    session.add_events(
        "emg_breaths",
        [BreathEvent("emg", 0.002, 0.004, peak_time=0.003)],
    )

    fig = plot_synchronization_comparison(session, max_seconds=None)

    try:
        (
            emg_before_ax,
            emg_after_ax,
            pressure_before_ax,
            pressure_after_ax,
            flow_before_ax,
            flow_after_ax,
            volume_before_ax,
            volume_after_ax,
        ) = fig.axes
        assert list(emg_before_ax.lines[0].get_ydata()) == [0.0, 0.0, 1.0, 2.0]
        assert list(emg_after_ax.lines[0].get_ydata()) == [1.0, 2.0]
        assert list(emg_after_ax.lines[1].get_xdata()) == [0.001, 0.001]
        assert emg_before_ax.get_title() == "EMG raw (EMG) before synchronization"
        assert emg_after_ax.get_title() == "EMG raw (EMG) after synchronization"
        assert list(pressure_before_ax.lines[0].get_ydata()) == [8.0, 9.0, 10.0]
        assert list(pressure_after_ax.lines[0].get_ydata()) == [9.0, 10.0]
        assert pressure_before_ax.get_title() == (
            "Ventilator pressure before synchronization"
        )
        assert pressure_after_ax.get_title() == (
            "Ventilator pressure after synchronization"
        )
        assert list(flow_before_ax.lines[0].get_ydata()) == [0.0, 0.5, 0.0]
        assert list(flow_after_ax.lines[0].get_ydata()) == [0.5, 0.0]
        assert flow_before_ax.get_title() == "Ventilator flow before synchronization"
        assert flow_after_ax.get_title() == "Ventilator flow after synchronization"
        assert list(volume_before_ax.lines[0].get_ydata()) == [100.0, 125.0, 150.0]
        assert list(volume_after_ax.lines[0].get_ydata()) == [125.0, 150.0]
        assert volume_before_ax.get_title() == (
            "Ventilator volume before synchronization"
        )
        assert volume_after_ax.get_title() == (
            "Ventilator volume after synchronization"
        )
    finally:
        plt.close(fig)


def test_emg_real_data_pipeline_uses_committed_poly5_sample():
    pytest.importorskip("resurfemg")

    repo_root = Path(__file__).resolve().parents[1]
    emg_path = os.path.join(
        repo_root,
        "data",
        "source",
        "data_from_repo",
        "emg_data_synth_quiet_breathing.Poly5",
    )
    vent_path = os.path.join(
        repo_root,
        "data",
        "source",
        "data_from_repo",
        "vent_data_synth_quiet_breathing.Poly5",
    )
    assert os.path.isfile(emg_path), f"missing fixture: {emg_path}"
    assert os.path.isfile(vent_path), f"missing fixture: {vent_path}"
    session = M3Session()

    session.load_emg(emg_path, verbose=False)

    assert session.emg is not None
    assert session.emg.raw is not None
    assert session.emg.dataframe is not None
    assert session.emg.metadata is not None
    assert session.emg.metadata["fs"] > 0
    assert session.emg.metadata["labels"]

    processed = session.preprocess_emg()

    assert len(processed["raw_channel"]) > 0
    assert len(processed["filtered"]) == len(processed["raw_channel"])
    assert len(processed["envelope"]) == len(processed["raw_channel"])
    assert session.emg.filtered is processed["filtered"]
    assert session.emg.envelope is processed["envelope"]
    assert session.emg.channel == 0
    assert session.emg.fs == processed["fs"]

    events = session.detect_emg_breaths()

    assert isinstance(events, list)
    assert all(isinstance(event, BreathEvent) for event in events)

    ventilator = session.emg_adapter.load(str(vent_path), verbose=False)
    postprocessing = session.postprocess_emg(ventilator=ventilator)

    assert "baseline" in postprocessing["available"]
    assert "moving_baseline" in postprocessing["computed"]["baseline"]
    assert "slopesum_baseline" in postprocessing["computed"]["baseline"]
    assert "find_occluded_breaths" in postprocessing["computed"]["event_detection"]
    assert "detect_ventilator_breath" in postprocessing["computed"]["event_detection"]
    assert (
        len(postprocessing["computed"]["event_detection"]["detect_ventilator_breath"])
        > 0
    )


def test_run_postprocessing_function_exposes_resurfemg_functions():
    pytest.importorskip("resurfemg")
    np = pytest.importorskip("numpy")

    adapter = ReSurfEMGAdapter()
    baseline = adapter.run_postprocessing_function(
        "baseline",
        "moving_baseline",
        np.asarray([0.0, 1.0, 0.0]),
        3,
        1,
    )

    assert len(baseline) == 3
    assert "quality_assessment" in adapter.available_postprocessing()
