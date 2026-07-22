"""Name -> `Pipeline` class registry (plan_stage2.md Sec 19)."""

from __future__ import annotations

from m3resp.core.exceptions import UnknownPipelineError
from m3resp.presets.base import Pipeline
from m3resp.presets.eit import EITPipeline
from m3resp.presets.emg import EMGPipeline
from m3resp.presets.multimodal import MultimodalPipeline

PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {}


def register_pipeline(name: str, pipeline_cls: type[Pipeline]) -> None:
    """Register ``pipeline_cls`` under ``name``."""

    PIPELINE_REGISTRY[name] = pipeline_cls


def get_pipeline(name: str) -> type[Pipeline]:
    """Return the registered `Pipeline` class for ``name``."""

    try:
        return PIPELINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PIPELINE_REGISTRY)) or "(none registered)"
        raise UnknownPipelineError(
            f"Unknown pipeline '{name}'. Available pipelines: {available}."
        ) from exc


def available_pipelines() -> list[str]:
    """Return the names of all registered pipelines, sorted."""

    return sorted(PIPELINE_REGISTRY)


register_pipeline("eit", EITPipeline)
register_pipeline("emg", EMGPipeline)
register_pipeline("multimodal", MultimodalPipeline)
