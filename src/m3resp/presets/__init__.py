"""Named EIT, EMG and multimodal presets for loaded sessions.

Presets run fixed operation sequences with configurable settings. Use
m3resp.workflows for custom YAML/JSON workflow descriptions.
"""

from __future__ import annotations

from m3resp.presets.base import Preset, PresetConfig
from m3resp.presets.eit import EITPreset
from m3resp.presets.emg import EMGPreset
from m3resp.presets.multimodal import MultimodalPreset
from m3resp.presets.registry import (
    PRESET_REGISTRY,
    available_presets,
    get_preset,
    register_preset,
)

__all__ = [
    "PRESET_REGISTRY",
    "EITPreset",
    "EMGPreset",
    "MultimodalPreset",
    "Preset",
    "PresetConfig",
    "available_presets",
    "get_preset",
    "register_preset",
]
