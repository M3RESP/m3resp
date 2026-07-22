"""Namespaced loader functions, grouped by modality."""

from m3resp.modalities.eit import load as load_eit
from m3resp.modalities.emg import load as load_emg

__all__ = ["load_eit", "load_emg"]
