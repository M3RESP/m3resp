"""Convenience workflow entry points."""

from m3resp.workflows.auto import run, run_workflow, select_workflow
from m3resp.workflows.eit_workflow import run_eit_workflow
from m3resp.workflows.emg_workflow import run_emg_workflow
from m3resp.workflows.multimodal_workflow import run_multimodal_workflow
from m3resp.workflows.rotarc_breath_duration import run_rotarc_breath_duration_workflow
from m3resp.workflows.configured import (
    WorkflowResult,
    save_workflow_figures,
    summarize_eit,
    summarize_emg,
    summarize_emg_postprocessing,
    summarize_multimodal,
)
from m3resp.workflows.toolbox import (
    configure_workflow_logging,
    configure_workflow_paths,
    find_repo_root,
    get_config_path,
    log_workflow_summary,
)

__all__ = [
    "WorkflowResult",
    "configure_workflow_logging",
    "configure_workflow_paths",
    "find_repo_root",
    "get_config_path",
    "log_workflow_summary",
    "run",
    "run_eit_workflow",
    "run_emg_workflow",
    "run_multimodal_workflow",
    "run_rotarc_breath_duration_workflow",
    "run_workflow",
    "save_workflow_figures",
    "select_workflow",
    "summarize_eit",
    "summarize_emg",
    "summarize_emg_postprocessing",
    "summarize_multimodal",
]
