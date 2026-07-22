"""`MultimodalPipeline`: the built-in "multimodal" preset (plan_stage2.md Sec 18)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Pipeline, PipelineConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class MultimodalPipeline(Pipeline):
    """Synchronize raw signals and align detected events across modalities.

    Equivalent to calling ``session.synchronize_raw_modalities()`` then
    ``session.align_modalities()`` directly; run after the per-modality
    pipelines so their breath events already exist.
    """

    name = "multimodal"

    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        session.synchronize_raw_modalities(
            **self._kwargs_for(config, "synchronize_raw")
        )
        session.align_modalities(**self._kwargs_for(config, "align"))
        return session
