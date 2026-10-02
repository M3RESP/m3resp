"""`MultimodalPreset`: the built-in "multimodal" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class MultimodalPreset(Preset):
    """Synchronize raw signals and align detected breath events across modalities.

    Equivalent to calling ``session.synchronize_raw_modalities()`` then
    ``session.synchronize_multimodal_breaths()`` directly; run after the
    per-modality presets so their breath events already exist.
    """

    name = "multimodal"

    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        session.synchronize_raw_modalities(
            **self._kwargs_for(config, "synchronize_raw")
        )
        session.synchronize_multimodal_breaths(**self._kwargs_for(config, "align"))
        return session
