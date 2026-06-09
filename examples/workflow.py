"""Run the M3Resp workflow selected by examples/config.yaml."""

from __future__ import annotations

import os

from m3resp import load_workflow_config
from m3resp.workflows.auto import run
from m3resp.workflows.toolbox import (
    configure_workflow_logging,
    configure_workflow_paths,
    get_config_path,
    log_workflow_summary,
)


REPO_ROOT = configure_workflow_paths("eitprocessing")
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "config.yaml")


def main() -> None:
    """Run the example and log a compact summary."""

    configure_workflow_logging()
    config_path = get_config_path(CONFIG_PATH)
    cfg = load_workflow_config(config_path)
    result = run(config_path)
    log_workflow_summary(
        "Configured workflow complete.",
        result.output_dir,
        result.summary,
        active_modules={
            "eit": cfg.modules.eit,
            "emg": cfg.modules.emg,
            "vent": cfg.modules.vent,
        },
    )


if __name__ == "__main__":
    main()
