"""`EMGPipeline`: the built-in "emg" preset (plan_stage2.md Sec 18)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.presets.base import Pipeline, PipelineConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EMGPipeline(Pipeline):
    """Preprocess, remove ECG, detect breaths, and postprocess loaded EMG data.

    Equivalent to calling ``session.preprocess_emg()``, the
    ``emg.ecg_detect_peaks`` + ``emg.ecg_gating`` steps,
    ``session.detect_emg_breaths()``, then ``session.postprocess_emg()``
    directly; expects ``session.load_emg(...)`` to have already been called.

    ECG removal by gating runs *by default*, between preprocessing and breath
    detection. ``session.preprocess_emg()`` only band-passes and envelopes the
    signal, and a band-pass high enough to suppress ECG (the ``high_pass_hz``
    default) leaves the higher-frequency part of each QRS complex inside the
    pass band - so an envelope computed straight off the filtered signal is
    ECG-contaminated. Gating each detected ECG peak and recomputing the
    envelope from the gated signal is the standard preprocessing chain
    (band-pass -> ECG peak detection -> gating -> envelope -> baseline), and
    this preset is what "the standard EMG pipeline" means, so it does that
    rather than leaving it to the caller.

    Config keys, each a mapping of keyword arguments:

    - ``preprocess`` -> ``session.preprocess_emg``
    - ``ecg_detect_peaks`` -> the ``emg.ecg_detect_peaks`` step (pass
      ``{"ecg_channel": n}`` when a dedicated reference ECG channel was
      recorded; the default detects peaks in the EMG channel itself)
    - ``ecg_gating`` -> the ``emg.ecg_gating`` step
    - ``detect_breaths`` -> ``session.detect_emg_breaths``
    - ``postprocess`` -> ``session.postprocess_emg``

    Plus one toggle: ``{"ecg_removal": {"enabled": False}}`` skips ECG removal
    entirely, leaving the pre-gating envelope in place. That is a data-check /
    exploratory path, not a scientifically valid default - the resulting
    envelope, breath detections, and every amplitude-derived parameter
    downstream of it still contain cardiac signal.
    """

    name = "emg"

    def run(
        self, session: M3Session, *, config: PipelineConfig | None = None
    ) -> M3Session:
        processed = session.preprocess_emg(**self._kwargs_for(config, "preprocess"))
        self._remove_ecg(session, processed, config)
        session.detect_emg_breaths(**self._kwargs_for(config, "detect_breaths"))
        session.postprocess_emg(**self._kwargs_for(config, "postprocess"))
        return session

    def _remove_ecg(
        self,
        session: M3Session,
        processed: Any,
        config: PipelineConfig | None,
    ) -> None:
        """Detect ECG peaks and gate them out, updating `session.processed`.

        The two registered steps are called as plain functions: they already
        record provenance through `M3Session._record()` and populate the typed
        collections themselves, so this stays a sequence of instrumented calls
        and does not need the declarative engine (see `presets.base`).
        """

        removal_options = dict((config or {}).get("ecg_removal", {}))
        if not removal_options.pop("enabled", True):
            return
        if removal_options:
            raise TypeError(
                "EMGPipeline config['ecg_removal'] only accepts 'enabled'; got "
                f"{sorted(removal_options)}. Step keyword arguments belong "
                "under config['ecg_detect_peaks'] / config['ecg_gating']."
            )

        # Imported here, not at module scope: the step modules import
        # `m3resp.core.session`, which would make this a circular import at
        # package-import time (mirrors `M3Session.run_pipeline`'s own lazy
        # import of the preset registry).
        from m3resp.workflows.steps.emg.ecg_detection import ecg_detect_peaks
        from m3resp.workflows.steps.emg.ecg_gating import ecg_gating

        detected = ecg_detect_peaks(
            session, processed, **self._kwargs_for(config, "ecg_detect_peaks")
        )
        # `or {}` only to satisfy `StepCallable`'s `Mapping | None` return
        # type; this step always returns its declared writes.
        ecg_gating(
            session,
            processed,
            (detected or {})["ecg_peak_indices"],
            **self._kwargs_for(config, "ecg_gating"),
        )
