"""Importing this package registers all built-in workflow steps.

Step modules import their upstream dependencies inside the step functions,
so importing this package never requires the optional ``resurfemg`` package.
"""

from m3resp.workflows.steps import (
    eit,
    emg,
    export,
    metrics,
    sync,
    ventilator,
)

__all__ = ["eit", "emg", "export", "metrics", "sync", "ventilator"]
