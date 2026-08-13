"""Execution engine for declarative M3Resp pipelines.

This package mirrors the former single ``engine.py`` module, split by
responsibility for readability: ``_shared.py`` (the `PipelineResult` type
and step-registration bookkeeping shared across the rest), ``execution.py``
(the compiled-pipeline run loop), ``spec_runner.py`` (spec loading, running,
and output/manifest writing), and ``diagnostics.py`` (spec validation and
compile-time diagnostics). Every public name is re-exported here so
``from m3resp.workflows.engine import <name>`` keeps working unchanged.
"""

from __future__ import annotations

from ._shared import PipelineResult
from .diagnostics import collect_diagnostics, validate_spec
from .execution import run_pipeline
from .spec_runner import run_spec

__all__ = [
    "PipelineResult",
    "collect_diagnostics",
    "run_pipeline",
    "run_spec",
    "validate_spec",
]
