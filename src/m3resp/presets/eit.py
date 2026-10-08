"""`EITPipeline`: the built-in "eit" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Pipeline, PipelineConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EITPipeline(Pipeline):
    """A starting point for an EIT pipeline, registered as the "eit" preset.

    ``run()`` calls ``session.preprocess_eit(**config["preprocess"])`` then
    ``session.detect_eit_breaths(**config["detect_breaths"])``. It expects
    ``session.load_eit(...)`` to have already been called. Most projects
    adjust it through ``config``, or subclass or replace it.
    ``preprocess_eit`` currently also runs breath detection and computes
    rates/TIV/EELI/pixel TIV internally, following the shape of
    ``eitprocessing``'s own API (see PR #23), so "preprocess then detect"
    means "run the adapter's default bundle, then normalize its breath
    output." Revisit this pipeline's shape once EIT is natively
    reimplemented (Stage 3).

    Every option ``preprocess_eit``/``detect_eit_breaths`` accept - filter
    mode (mdn, lowpass, bandpass, none), which optional outputs to compute
    (rates, TIV, EELI, pixel TIV), breath-detection parameters - is still
    reachable through ``config``. ``EITPipeline()`` with no config runs those
    methods' own defaults.

    It gives the common "run default EIT processing" case a name
    (``session.run_pipeline("eit", config=...)``) that's swappable with
    ``"emg"``/``"multimodal"`` through the same registry (``get_pipeline``/
    ``available_pipelines``) - useful for code that picks a modality's
    pipeline generically, or as a documented starting point for a project's
    own custom preset. See `m3resp.presets.base` for how presets relate to
    the declarative YAML/JSON engine in `m3resp.workflows`.
    """

    name = "eit"

    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        session.preprocess_eit(**self._kwargs_for(config, "preprocess"))
        session.detect_eit_breaths(**self._kwargs_for(config, "detect_breaths"))
        return session
