"""Importing this package registers all built-in pipeline steps.

Step modules import their upstream dependencies lazily (inside the step
functions), so importing this package never requires the optional ``eitprocessing``
or ``resurfemg`` packages.
"""

from m3resp.workflows.steps import (  # noqa: F401
    eit,
    emg,
    export,
    metrics,
    session,
    sync,
)

__all__ = ["eit", "emg", "export", "metrics", "session", "sync"]
