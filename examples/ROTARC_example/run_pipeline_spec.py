"""Run the ROTARC breath-duration variability pipeline from a declarative spec.

This is the spec-driven equivalent of ``breath-duration.py``: instead of calling
a workflow function, it executes ``breath-duration.pipeline.yaml`` directly
through the generic pipeline engine, demonstrating that the breath-detection
pipeline is reconstructed from the spec with no custom Python.
"""

from __future__ import annotations

import os

from loguru import logger

from m3resp import run_pipeline
from m3resp.workflows.toolbox import (
    configure_workflow_logging,
    configure_workflow_paths,
)

REPO_ROOT = configure_workflow_paths("eitprocessing")
SPEC_PATH = os.path.join(
    str(REPO_ROOT), "examples", "ROTARC_example", "breath-duration.pipeline.yaml"
)


def main() -> None:
    """Execute the breath-duration pipeline spec and log the CV result."""

    configure_workflow_logging()
    result = run_pipeline(SPEC_PATH)

    logger.success("ROTARC breath-duration pipeline complete.")
    logger.info("Breaths: {}", result.value("n"))
    logger.info("Breath duration CV: {:.8f}", result.value("breath_duration_cv"))


if __name__ == "__main__":
    main()
