"""`EMGPipeline`: the built-in "emg" preset (plan_stage2.md Sec 18)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Pipeline, PipelineConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EMGPipeline(Pipeline):
    """Preprocess, detect breaths, and postprocess already-loaded EMG data.

    Equivalent to calling ``session.preprocess_emg()``,
    ``session.detect_emg_breaths()``, then ``session.postprocess_emg()``
    directly; expects ``session.load_emg(...)`` to have already been called.
    """

    name = "emg"

    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        session.preprocess_emg(**self._kwargs_for(config, "preprocess"))
        session.detect_emg_breaths(**self._kwargs_for(config, "detect_breaths"))
        session.postprocess_emg(**self._kwargs_for(config, "postprocess"))
        return session
