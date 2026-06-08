"""Convenience workflow entry points."""

from m3resp.workflows.eit_workflow import run_eit_workflow
from m3resp.workflows.emg_workflow import run_emg_workflow
from m3resp.workflows.multimodal_workflow import run_multimodal_workflow

__all__ = ["run_eit_workflow", "run_emg_workflow", "run_multimodal_workflow"]
