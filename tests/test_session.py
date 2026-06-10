import os
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

    assert synchronized["eit_breaths"][0].start_time == 1.0
    assert session.parameters["alignment"]["reference_modality"] == "eit"
    assert session.parameters["alignment"]["fallback_reference_modality"] == "eit"
    assert os.path.exists(os.path.join(output_dir, "summary.json"))
    assert os.path.exists(os.path.join(output_dir, "eit_breaths.csv"))


def test_session_event_helpers_keep_events_dict_as_backing_store():
    session = M3Session()
    events = [BreathEvent("emg", 0.0, 1.0, peak_time=0.5)]

    stored = session.add_events("emg_breaths", events)

    assert stored == events
    assert session.events["emg_breaths"] == events
    assert session.get_events("emg_breaths") == events
    assert session.get_events("missing", default=[]) == []


def test_session_normalizes_ventilator_breaths_after_emg_postprocessing():
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda *args, **kwargs: {"emg": True})
    )
    session.processed["emg"] = {"filtered": True}

    def postprocess(data, events=None, **kwargs):
        return {
            "available": {"event_detection": ["detect_ventilator_breath"]},
            "computed": {"event_detection": {"detect_ventilator_breath": [100, 200]}},
            "skipped": {},
        }

    session.emg_adapter.postprocess = postprocess
    session.postprocess_emg(
        ventilator={"metadata": {"fs": 100.0}, "array": [[0.0, 1.0]]},
        ventilator_breath_width_seconds=0.4,
    )

    breaths = session.events["ventilator_breaths"]
    assert [breath.modality for breath in breaths] == ["vent", "vent"]
    assert breaths[0].peak_time == 1.0
    assert breaths[0].start_time == 0.8
    assert breaths[0].end_time == 1.2
    assert breaths[0].metadata["fs"] == 100.0


def test_session_aligns_eit_emg_and_ventilator_events_with_offset_map():
    session = M3Session()
    eit_events = [BreathEvent("eit", 1.0, 2.0, peak_time=1.5)]
    emg_events = [BreathEvent("emg", 1.0, 2.0, peak_time=1.5)]
    vent_events = [BreathEvent("vent", 1.0, 2.0, peak_time=1.5)]
    session.add_events("eit_breaths", eit_events)
    session.add_events("emg_breaths", emg_events)
    session.add_events("ventilator_breaths", vent_events)

    synchronized = session.align_modalities(
        offset_seconds={"eit": -0.1, "emg": 0.25, "vent": 0.0}
    )

    assert synchronized["eit_breaths"][0].start_time == 0.9
    assert synchronized["emg_breaths"][0].start_time == 1.25
    assert synchronized["ventilator_breaths"][0].start_time == 1.0
    assert session.events["emg_breaths"][0].start_time == 1.0
    assert session.parameters["alignment"]["reference_modality"] == "vent"
    assert session.parameters["alignment"]["fallback_reference_modality"] is None
    assert session.parameters["alignment"]["offset_seconds"] == {
        "eit": -0.1,
        "emg": 0.25,
        "vent": 0.0,
    }
    assert session.provenance[-1].action == "align_modalities"


def test_session_scalar_alignment_offsets_emg_only_and_rejects_unknown_method():
    session = M3Session()
    session.add_events("eit_breaths", [BreathEvent("eit", 1.0, 2.0)])
    session.add_events("emg_breaths", [BreathEvent("emg", 1.0, 2.0)])
    session.add_events("ventilator_breaths", [BreathEvent("vent", 1.0, 2.0)])

    synchronized = session.align_modalities(offset_seconds=0.5)

    assert synchronized["eit_breaths"][0].start_time == 1.0
    assert synchronized["emg_breaths"][0].start_time == 1.5
    assert synchronized["ventilator_breaths"][0].start_time == 1.0

    try:
        session.align_modalities(method="timestamp")
    except ValueError as exc:
        assert "manual_offset" in str(exc)
    else:
        raise AssertionError(
            "Expected unsupported alignment method to raise ValueError"
        )
