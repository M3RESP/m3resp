"""Deprecated alias for :mod:`m3resp.workflows.spec`.

See :mod:`m3resp.pipeline` for the deprecation notice and removal target.
"""

from __future__ import annotations

import warnings

from m3resp.workflows.spec import (
    PipelineSpec,
    SpecExperimentConfig,
    SpecOutputsConfig,
    StepSpec,
    load_spec,
)

warnings.warn(
    "m3resp.pipeline.spec is deprecated and will be removed no earlier "
    "than m3resp 0.3.0; import from m3resp.workflows.spec instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PipelineSpec",
    "SpecExperimentConfig",
    "SpecOutputsConfig",
    "StepSpec",
    "load_spec",
]
