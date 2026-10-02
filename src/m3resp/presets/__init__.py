"""Named, built-in presets: short, fixed sequences of `M3Session` calls.

Distinct from ``m3resp.workflows``, the declarative step-registry engine used
for fully custom YAML/JSON workflows. See ``base.py``'s module docstring and
``plan/stage2_consolidation.md`` for how the two relate.
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
