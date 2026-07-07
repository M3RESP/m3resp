"""Named, built-in `Pipeline` presets (plan_stage2.md Sec 18-19, Milestone 2.4).

Distinct from ``m3resp.workflows``, the declarative step-registry engine used
for fully custom YAML/JSON workflows. See ``base.py``'s module docstring and
``plan/stage2_consolidation.md`` for how the two relate.
"""

from __future__ import annotations

from m3resp.presets.base import Pipeline, PipelineConfig
from m3resp.presets.eit import EITPipeline
from m3resp.presets.emg import EMGPipeline
from m3resp.presets.multimodal import MultimodalPipeline
from m3resp.presets.registry import (
    PIPELINE_REGISTRY,
    available_pipelines,
    get_pipeline,
    register_pipeline,
)

__all__ = [
    "PIPELINE_REGISTRY",
    "EITPipeline",
    "EMGPipeline",
    "MultimodalPipeline",
    "Pipeline",
    "PipelineConfig",
    "available_pipelines",
    "get_pipeline",
    "register_pipeline",
]
