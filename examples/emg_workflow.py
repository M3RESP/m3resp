"""Minimal EMG workflow with injected loader and preprocessing."""

from m3resp import M3Session
from m3resp.adapters import ReSurfEMGAdapter


def load_emg(path: str, **kwargs):
    return {"path": path, "samples": [0.1, 0.3, 0.2]}


def preprocess(signal, **kwargs):
    return {**signal, "filtered": True}


session = M3Session(emg_adapter=ReSurfEMGAdapter(loader=load_emg))
session.load_emg("example.edf")
session.preprocess_emg(preprocess=preprocess)

print(session.processed["emg"])
