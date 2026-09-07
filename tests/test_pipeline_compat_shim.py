"""Tests for the deprecated ``m3resp.pipeline`` compatibility shim.

See ``plan/stage2/3_pipeline_structure_implementation_plan.md`` Phase 0.1:
the shim re-exports the ``m3resp.workflows`` API under the earlier import
path and warns, but must not become a second implementation.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Any

import pytest

_SHIM_MODULES = [
    "m3resp.pipeline",
    "m3resp.pipeline.engine",
    "m3resp.pipeline.registry",
    "m3resp.pipeline.spec",
    "m3resp.pipeline.context",
]


@pytest.fixture(autouse=True)
def _fresh_shim_imports():
    """Each test re-imports the shim modules so the deprecation warning
    fires every time, instead of only on the first import of the process."""

    for name in _SHIM_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in _SHIM_MODULES:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("module_name", _SHIM_MODULES)
def test_shim_module_warns_on_import(module_name: str):
    with pytest.warns(DeprecationWarning, match="m3resp.workflows"):
        importlib.import_module(module_name)


def test_pipeline_package_reexports_workflows_api():
    import m3resp.pipeline as legacy
    import m3resp.workflows as current

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(current, name)


def test_pipeline_engine_reexports_workflows_engine():
    import m3resp.pipeline.engine as legacy
    from m3resp.workflows import engine as current

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(current, name)


def test_pipeline_registry_reexports_workflows_registry():
    import m3resp.pipeline.registry as legacy
    from m3resp.workflows import registry as current

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(current, name)


def test_pipeline_spec_reexports_workflows_spec():
    import m3resp.pipeline.spec as legacy
    from m3resp.workflows import spec as current

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(current, name)


def test_pipeline_context_reexports_workflows_context():
    import m3resp.pipeline.context as legacy
    from m3resp.workflows import context as current

    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(current, name)


def test_legacy_run_pipeline_behaves_like_current_run_pipeline():
    """The shim must not diverge into a second implementation: running the
    same spec through both import paths must produce the same result."""

    from m3resp.workflows.registry import STEP_REGISTRY, register_step

    @register_step("shim_compat.make", writes=("value",))
    def _make(*, x: int) -> dict[str, Any]:
        return {"value": x * 3}

    spec = {
        "name": "shim-smoke",
        "inputs": {"x": 7},
        "steps": [{"uses": "shim_compat.make", "with": {"x": "@x"}}],
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import m3resp.pipeline as legacy

            legacy_result = legacy.run_pipeline(spec)

        from m3resp.workflows import run_pipeline as current_run_pipeline

        current_result = current_run_pipeline(spec)

        assert legacy_result.value("value") == current_result.value("value") == 21
    finally:
        STEP_REGISTRY.pop("shim_compat.make", None)
