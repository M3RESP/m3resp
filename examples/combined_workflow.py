"""Minimal combined workflow with upstream integrations."""

from m3resp import M3Session
from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter


def load_eit(path: str, vendor: str | None = None, **kwargs):
    return {"path": path, "vendor": vendor}


def load_emg(path: str, **kwargs):
    return {"path": path}


def breath_detector(data, **kwargs):
    return [{"start_time": 0.0, "end_time": 1.0, "peak_time": 0.5}]


session = M3Session(
    eit_adapter=EITProcessingAdapter(loader=load_eit),
    emg_adapter=ReSurfEMGAdapter(loader=load_emg),
)
session.load_eit("example.eit", vendor="sentec")
session.load_emg("example.edf")
session.preprocess_eit()
session.detect_eit_breaths(detector=breath_detector)
session.detect_emg_breaths(detector=breath_detector)
session.align_modalities(offset_seconds=0.25)
session.export_summary("output/example-summary")
