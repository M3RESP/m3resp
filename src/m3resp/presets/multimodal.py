"""`MultimodalPreset`: the built-in "multimodal" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class MultimodalPreset(Preset):
    """Set recording start times and align detected breaths across modalities.

    Runs ``session.synchronize_raw_modalities`` followed by
    ``session.synchronize_multimodal_breaths``. Run after processing each
    modality so the breath events are available.

    Attributes:
        name: Registered preset name, ``"multimodal"``.
    """

    name = "multimodal"

    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        """Set recording start times and store aligned breath events.

        Args:
            session: Session with recordings and detected breaths to synchronize.
            config: Keyword arguments for the session methods, grouped under
                ``synchronize_raw`` and ``align``. Offsets are in seconds; None
                uses the session methods' defaults.

        Returns:
            M3Session: The supplied session with start times, synchronization
                methods, aligned events and provenance.
        """

        session.synchronize_raw_modalities(
            **self._kwargs_for(config, "synchronize_raw")
        )
        session.synchronize_multimodal_breaths(**self._kwargs_for(config, "align"))
        return session
