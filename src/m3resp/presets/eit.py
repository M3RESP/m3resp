"""`EITPreset`: the built-in "eit" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EITPreset(Preset):
    """A starting point for an EIT workflow, registered as the "eit" preset -
    not "the" canonical EIT algorithm.

    What it actually does, concretely: ``run()`` calls
    ``session.preprocess_eit(**config["preprocess"])`` then
    ``session.detect_eit_breaths(**config["detect_breaths"])`` - two lines,
    nothing else. It expects ``session.load_eit(...)`` to have already been
    called. It is deliberately thin, and that's expected to mean most
    projects amend it via ``config`` (or subclass/replace it) rather than use
    it unmodified - see ``m3resp.presets.base`` for why: ``preprocess_eit``
    currently also runs breath detection and computes rates/TIV/EELI/pixel
    TIV internally (an artifact of wrapping ``eitprocessing``'s own API
    shape, not an m3resp design choice - see PR #23 discussion), so what
    reads as "preprocess then detect" is really "run the adapter's default
    bundle, then normalize its breath output." Revisit this preset's shape
    once EIT is natively reimplemented (Stage 3) and that bundling can be
    restructured.

    Every option ``preprocess_eit``/``detect_eit_breaths`` accept - filter
    mode (mdn, lowpass, bandpass, none), which optional outputs to compute
    (rates, TIV, EELI, pixel TIV), breath-detection parameters - is still
    reachable through ``config``. ``EITPreset()`` with no config runs those
    methods' own defaults; it doesn't hide or replace any choice.

    Why it exists rather than just calling those two methods directly: it
    gives the common "run default EIT processing" case a name
    (``session.run_preset("eit", config=...)``) that's swappable with
    ``"emg"``/``"multimodal"`` through the same registry (``get_preset``/
    ``available_presets``) - useful for code that picks a modality's
    preset generically, or as a documented starting point for a project's
    own custom preset. See `m3resp.presets.base` for why this is a separate,
    deliberately non-general mechanism from the declarative YAML/JSON engine
    in `m3resp.workflows`.
    """

    name = "eit"

    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        session.preprocess_eit(**self._kwargs_for(config, "preprocess"))
        session.detect_eit_breaths(**self._kwargs_for(config, "detect_breaths"))
        return session
