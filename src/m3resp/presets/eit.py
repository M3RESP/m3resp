"""`EITPreset`: the built-in "eit" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EITPreset(Preset):
    """Preprocess loaded EIT data and store its detected breaths.

    Runs ``session.preprocess_eit`` followed by ``session.detect_eit_breaths``.
    The preprocessing (``m3resp.eit.processing.preprocess_eit_recording``)
    also computes breath results, such as rates, TIV and EELI, when the
    options ask for them.

    Attributes:
        name: Registered preset name, ``"eit"``.
    """

    name = "eit"

    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        """Preprocess EIT data and store detected breaths on the session.

        Args:
            session: Session with EIT data already loaded.
            config: Keyword arguments grouped under ``preprocess`` and
                ``detect_breaths``. None uses the session methods' defaults.

        Returns:
            M3Session: The supplied session with EIT processing results, breath
                events and provenance.

        Raises:
            MissingModalityDataError: If the required EIT recording is unavailable.
        """

        session.preprocess_eit(**self._kwargs_for(config, "preprocess"))
        session.detect_eit_breaths(**self._kwargs_for(config, "detect_breaths"))
        return session
