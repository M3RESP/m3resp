"""Minimal EIT workflow with an injected loader for local development."""

from m3resp import M3Session
from m3resp.adapters import EITProcessingAdapter


def load_eit(path: str, vendor: str | None = None, **kwargs):
    return {"path": path, "vendor": vendor, "samples": [1.0, 2.0, 1.5]}


session = M3Session(eit_adapter=EITProcessingAdapter(loader=load_eit))
session.load_eit("example.eit", vendor="sentec")
session.preprocess_eit()

print(session.raw["eit"])
