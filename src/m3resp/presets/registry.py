"""Name -> `Preset` class registry."""

from __future__ import annotations

from m3resp.core.exceptions import UnknownPresetError
from m3resp.presets.base import Preset
from m3resp.presets.eit import EITPreset
from m3resp.presets.emg import EMGPreset
from m3resp.presets.multimodal import MultimodalPreset

PRESET_REGISTRY: dict[str, type[Preset]] = {}


def register_preset(name: str, preset_cls: type[Preset]) -> None:
    """Store preset_cls under name, replacing any existing registration."""

    PRESET_REGISTRY[name] = preset_cls


def get_preset(name: str) -> type[Preset]:
    """Return the Preset class registered under name.

    Raises UnknownPresetError if name is unregistered, listing available names.
    """

    try:
        return PRESET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESET_REGISTRY)) or "(none registered)"
        raise UnknownPresetError(
            f"Unknown preset '{name}'. Available presets: {available}."
        ) from exc


def available_presets() -> list[str]:
    """Return the names of all registered presets, sorted."""

    return sorted(PRESET_REGISTRY)


register_preset("eit", EITPreset)
register_preset("emg", EMGPreset)
register_preset("multimodal", MultimodalPreset)
