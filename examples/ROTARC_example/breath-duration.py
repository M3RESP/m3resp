"""Run the ROTARC breath-duration variability workflow from YAML config."""

from __future__ import annotations

import os

from loguru import logger

from m3resp.workflows import run_rotarc_breath_duration_workflow
from m3resp.workflows.toolbox import (
    configure_workflow_logging,
    configure_workflow_paths,
)


REPO_ROOT = configure_workflow_paths("eitprocessing")
CONFIG_PATH = os.path.join(str(REPO_ROOT), "examples", "ROTARC_example", "config.yaml")


def main() -> None:
    """Run the ROTARC breath-duration CV calculation configured in YAML."""

    configure_workflow_logging()
    result = run_rotarc_breath_duration_workflow(CONFIG_PATH)

    logger.success("ROTARC breath-duration workflow complete.")
    logger.info("Output directory: {}", result.output_dir)
    logger.info("Result path: {}", result.summary["result_path"])
    logger.info("Breaths: {}", result.summary["n_breaths"])
    logger.info(
        "Breath duration CV: {:.8f}",
        result.summary["breath_duration_cv"],
    )


if __name__ == "__main__":
    main()
