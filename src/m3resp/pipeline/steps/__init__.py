"""Importing this package registers all built-in pipeline steps.

Step modules import their upstream dependencies lazily (inside the step
functions), so importing this package never requires the optional ``eitprocessing``
or ``resurfemg`` packages.
"""

from m3resp.pipeline.steps import eit, emg, export, metrics, session  # noqa: F401

__all__ = ["eit", "emg", "export", "metrics", "session"]
