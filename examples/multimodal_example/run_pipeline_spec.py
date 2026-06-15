"""Run the multimodal workflow from a declarative pipeline spec.

This is the spec-driven equivalent of ``workflow.py``.  Instead of calling
``auto.run(config_path)``, it:

1. Compiles ``config.yaml`` into a declarative spec with
   :func:`~m3resp.pipeline.compile_config.build_multimodal_spec`.
2. Executes it through :func:`m3resp.run_pipeline` (the generic engine).
3. Assembles ``session.processed["eit"]`` so the same summary helpers work.
4. Logs the same compact summary as the original script.

Running this script against the same config and data should produce the same
output files and logged values as ``workflow.py``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from loguru import logger

from m3resp import load_workflow_config, run_pipeline
from m3resp.pipeline.compile_config import (
    build_multimodal_spec,
    build_eit_processing_plan,
)
from m3resp.workflows.configured.artifacts import export_configured_session
from m3resp.workflows.configured.runner import select_configured_workflow
from m3resp.workflows.configured.steps import assemble_eit_processed
from m3resp.workflows.configured.summaries import summarize_multimodal
from m3resp.workflows.toolbox import (
    configure_workflow_logging,
    configure_workflow_paths,
    get_config_path,
    log_workflow_summary,
)


REPO_ROOT = configure_workflow_paths("eitprocessing")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "multimodal_example", "config.yaml")


def main() -> None:
    """Run the multimodal pipeline spec and log a compact summary."""

    configure_workflow_logging()
    config_path = get_config_path(CONFIG_PATH)
    cfg = load_workflow_config(config_path)

    spec = build_multimodal_spec(cfg)
    result = run_pipeline(spec)

    session = result.session

    # Assemble session.processed["eit"] so the standard summary helpers work.
    if cfg.modules.eit and cfg.eit.processing.preprocess.enabled:
        plan = build_eit_processing_plan(cfg)
        seed = {
            "eit_sequence": result.value("eit_sequence"),
            "raw_eit": result.value("raw_eit"),
            "raw_global_impedance": result.value("raw_global_impedance"),
        }
        session.processed["eit"] = assemble_eit_processed(
            plan, seed, result.context.values
        )

    summary = summarize_multimodal(
        session,
        include_eit=cfg.modules.eit,
        include_emg=cfg.modules.emg,
    )

    selected = select_configured_workflow(cfg)
    run_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.path.join(str(cfg.output.combined), run_folder))
    export_configured_session(session, output_dir, cfg)

    log_workflow_summary(
        "Pipeline spec workflow complete.",
        output_dir,
        summary,
        active_modules={
            "eit": cfg.modules.eit,
            "emg": cfg.modules.emg,
            "vent": cfg.modules.vent,
        },
    )
    logger.info("Workflow type: {}", selected)
    logger.info("Output directory: {}", output_dir)


if __name__ == "__main__":
    main()
