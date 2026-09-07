"""Deprecated alias for :mod:`m3resp.workflows.engine`.

See :mod:`m3resp.pipeline` for the deprecation notice and removal target.
"""

from __future__ import annotations

import warnings

from m3resp.workflows.engine import (
    PipelineResult,
    run_pipeline,
    run_spec,
    validate_spec,
)

warnings.warn(
    "m3resp.pipeline.engine is deprecated and will be removed no earlier "
    "than m3resp 0.3.0; import from m3resp.workflows.engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PipelineResult", "run_pipeline", "run_spec", "validate_spec"]
