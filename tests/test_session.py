from pathlib import Path

from m3resp import BreathEvent, M3Session
from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter


def test_session_loads_modalities_with_injected_adapters():
    session = M3Session(
        eit_adapter=EITProcessingAdapter(
            loader=lambda path, vendor=None, **kwargs: {"path": path, "vendor": vendor}
        ),
        emg_adapter=ReSurfEMGAdapter(
            loader=lambda path, **kwargs: {"path": path, "kind": "emg"}
        ),
    )

    session.load_eit("subject.eit", vendor="sentec")
    session.load_emg("subject.edf")

    assert session.raw["eit"].vendor == "sentec"
    assert session.raw["emg"].path == Path("subject.edf")
    assert [entry.action for entry in session.provenance] == ["load_eit", "load_emg"]


def test_detection_alignment_and_export(tmp_path):
    session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda *args, **kwargs: {"eit": True}),
        emg_adapter=ReSurfEMGAdapter(loader=lambda *args, **kwargs: {"emg": True}),
    )
    session.load_eit("subject.eit", vendor="sentec")
    session.load_emg("subject.edf")
    session.processed["emg"] = {"filtered": True}

    def detector(data, **kwargs):
        return [BreathEvent("eit", 1.0, 2.0, peak_time=1.5)]

    session.detect_eit_breaths(detector=detector)
    synchronized = session.align_modalities(offset_seconds=0.5)
    output_dir = session.export_summary(tmp_path)

    assert synchronized["eit_breaths"][0].start_time == 1.5
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "eit_breaths.csv").exists()
