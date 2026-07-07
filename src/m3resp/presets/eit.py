"""`EITPipeline`: the built-in "eit" preset (plan_stage2.md Sec 18)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Pipeline, PipelineConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EITPipeline(Pipeline):
    """Preprocess and detect breaths for already-loaded EIT data.

    Equivalent to calling ``session.preprocess_eit(**config["preprocess"])``
    then ``session.detect_eit_breaths(**config["detect_breaths"])`` directly;
    expects ``session.load_eit(...)`` to have already been called.
    """

    name = "eit"

    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        session.preprocess_eit(**self._kwargs_for(config, "preprocess"))
        session.detect_eit_breaths(**self._kwargs_for(config, "detect_breaths"))
        return session
