"""Deprecated alias for :mod:`m3resp.workflows.context`.

See :mod:`m3resp.pipeline` for the deprecation notice and removal target.
"""

from __future__ import annotations

import warnings

from m3resp.workflows.context import SESSION_KEY, PipelineContext

warnings.warn(
    "m3resp.pipeline.context is deprecated and will be removed no earlier "
    "than m3resp 0.3.0; import from m3resp.workflows.context instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SESSION_KEY", "PipelineContext"]
