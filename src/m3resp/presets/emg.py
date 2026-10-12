"""`EMGPreset`: the built-in "emg" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


#: Settings the preset passes to `M3Session.emg_moving_baseline`.
_BASELINE_DEFAULTS = {"window_seconds": 7.5, "step_seconds": 0.2, "percentile": 33.0}


class EMGPreset(Preset):
    """Preprocess loaded EMG, remove ECG, detect breaths and calculate features.

    Runs preprocessing, ECG peak detection and gating, the envelope of the
    gated signal, a moving baseline, breath detection and postprocessing in
    that order.

    Config groups keyword arguments under ``preprocess``, ``ecg_detect_peaks``,
    ``ecg_gating``, ``envelope``, ``baseline``, ``detect_breaths`` and
    ``postprocess``. Envelope options are ``window_seconds`` and ``method``;
    they default to the envelope settings preprocessing used.
    Baseline options are ``window_seconds`` (7.5 s by default), ``step_seconds``
    (0.2 s) and ``percentile`` (33); other keys raise TypeError.
    ``ecg_detect_peaks.ecg_channel`` selects a reference ECG channel.

    ``ecg_removal.enabled=False`` skips gating; cardiac activity can then
    remain in the envelope and derived measurements. Supplying
    ``ecg_removal.ecg_peak_indices`` gates known sample positions and skips
    peak detection. Supplied peaks and nonempty ``ecg_detect_peaks`` options
    are mutually exclusive.

    Attributes:
        name: Registered preset name, ``"emg"``.
    """

    name = "emg"

    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        """Process loaded EMG and store breath events and measurements.

        Args:
            session: Session with EMG data already loaded.
            config: Settings grouped as described in `EMGPreset`. None uses the
                operation defaults, including ECG gating and moving-baseline
                estimation before breath detection.

        Returns:
            M3Session: The supplied session with processed signals, detected
                breaths, parameters, quality flags and provenance.

        Raises:
            MissingModalityDataError: If the required EMG recording is unavailable.
            TypeError: If ECG-removal options are unsupported or supplied ECG
                peaks conflict with peak-detection options.
        """

        processed = session.preprocess_emg(**self._kwargs_for(config, "preprocess"))
        self._remove_ecg(session, processed, config)
        baseline = self._moving_baseline(session, config)
        session.detect_emg_breaths(
            baseline=baseline, **self._kwargs_for(config, "detect_breaths")
        )
        session.postprocess_emg(**self._kwargs_for(config, "postprocess"))
        return session

    def _moving_baseline(self, session: M3Session, config: PresetConfig | None) -> Any:
        """Estimate the quiet level of the stored EMG envelope.

        Uses `M3Session.emg_moving_baseline` with config's ``baseline``
        options (``window_seconds``, ``step_seconds``, ``percentile``, each
        converted to a number). Other keys raise TypeError.
        Returns one value per envelope sample, in envelope units, or None when
        an envelope is unavailable.
        """

        processed = session.processed.get("emg") or {}
        if processed.get("envelope") is None:
            return None
        options = self._kwargs_for(config, "baseline")
        unknown = sorted(set(options) - set(_BASELINE_DEFAULTS))
        if unknown:
            raise TypeError(
                f"EMGPreset config['baseline'] accepts {sorted(_BASELINE_DEFAULTS)}; "
                f"got {unknown}. The baseline is computed from the same processed "
                "EMG that breath detection uses."
            )
        settings = {
            name: float(options.get(name, default))
            for name, default in _BASELINE_DEFAULTS.items()
        }
        return session.emg_moving_baseline(
            window_seconds=settings["window_seconds"],
            step_seconds=settings["step_seconds"],
            percentile=settings["percentile"],
        )

    def _remove_ecg(
        self,
        session: M3Session,
        processed: Any,
        config: PresetConfig | None,
    ) -> None:
        """Gate ECG peaks, then compute the envelope of the gated signal.

        Read ECG detection, gating and removal options from config. Use supplied
        ECG sample positions when available; otherwise detect them with the
        registered step. Returns immediately when removal is disabled.
        Raises TypeError for unsupported removal options or conflicting supplied
        peaks and detection options.
        """

        removal_options = dict((config or {}).get("ecg_removal", {}))
        if not removal_options.pop("enabled", True):
            return
        supplied_peak_indices = removal_options.pop("ecg_peak_indices", None)
        if removal_options:
            raise TypeError(
                "EMGPreset config['ecg_removal'] only accepts 'enabled' and "
                f"'ecg_peak_indices'; got {sorted(removal_options)}. Step "
                "keyword arguments belong under config['ecg_detect_peaks'] / "
                "config['ecg_gating']."
            )
        detection_kwargs = self._kwargs_for(config, "ecg_detect_peaks")
        if supplied_peak_indices is not None and detection_kwargs:
            raise TypeError(
                "EMGPreset: config['ecg_removal']['ecg_peak_indices'] skips "
                "peak detection, so config['ecg_detect_peaks'] "
                f"({sorted(detection_kwargs)}) would have no effect. Pass one "
                "or the other."
            )

        # Imported here, not at module scope: the step modules import
        # `m3resp.core.session`, which would make this a circular import at
        # package-import time (mirrors `M3Session.run_preset`'s own lazy
        # import of the preset registry).
        from m3resp.workflows.steps.emg.ecg_detection import ecg_detect_peaks
        from m3resp.workflows.steps.emg.ecg_gating import ecg_gating

        if supplied_peak_indices is not None:
            peak_indices: Any = supplied_peak_indices
        else:
            detected = ecg_detect_peaks(session, processed, **detection_kwargs)
            # `or {}` only to satisfy `StepCallable`'s `Mapping | None` return
            # type; this step always returns its declared writes.
            peak_indices = (detected or {})["ecg_peak_indices"]

        gated = ecg_gating(
            session,
            processed,
            peak_indices,
            **self._kwargs_for(config, "ecg_gating"),
        )
        # `or {}` only to satisfy `StepCallable`'s `Mapping | None` return type.
        self._envelope(
            session, processed, (gated or {})["processed_emg_after_ecg"], config
        )

    def _envelope(
        self,
        session: M3Session,
        processed: Any,
        gated: Any,
        config: PresetConfig | None,
    ) -> None:
        """Compute the envelope of the ECG-cleaned EMG bundle `gated`.

        Window and method come from config's ``envelope`` group, else from the
        envelope settings preprocessing recorded in `processed`.
        """

        from m3resp.workflows.steps.emg.preprocessing import envelope

        settings = (
            (processed.get("filter") or {}) if isinstance(processed, dict) else {}
        )
        kwargs = {
            "window_seconds": settings.get("envelope_window_seconds", 0.25),
            "method": settings.get("envelope_method") or "rms",
            **self._kwargs_for(config, "envelope"),
        }
        envelope(session, gated, **kwargs)
