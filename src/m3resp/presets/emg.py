"""`EMGPreset`: the built-in "emg" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


class EMGPreset(Preset):
    """Preprocess loaded EMG, remove ECG, detect breaths and calculate features.

    Runs preprocessing, ECG peak detection and gating, a moving baseline,
    breath detection and postprocessing in that order. Gating recomputes the
    EMG envelope before baseline estimation and breath detection.

    Config groups keyword arguments under ``preprocess``, ``ecg_detect_peaks``,
    ``ecg_gating``, ``baseline``, ``detect_breaths`` and ``postprocess``.
    Baseline options are ``window_seconds`` (7.5 s by default), ``step_seconds``
    (0.2 s) and additional moving-baseline options such as ``percentile``.
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

        Config's baseline window_seconds and step_seconds are converted using
        fs in Hz to at least one sample each; other options reach the adapter.
        Returns one value per envelope sample, in envelope units, or None when
        an envelope is unavailable.
        """

        import numpy as np

        processed = session.processed.get("emg") or {}
        envelope = processed.get("envelope")
        if envelope is None:
            return None

        kwargs = dict(self._kwargs_for(config, "baseline"))
        fs = float(processed["fs"])
        window_seconds = float(kwargs.pop("window_seconds", 7.5))
        step_seconds = float(kwargs.pop("step_seconds", 0.2))
        return session.emg_adapter.moving_baseline(
            np.asarray(envelope, dtype=float),
            window_samples=max(1, int(window_seconds * fs)),
            step_samples=max(1, int(step_seconds * fs)),
            **kwargs,
        )

    def _remove_ecg(
        self,
        session: M3Session,
        processed: Any,
        config: PresetConfig | None,
    ) -> None:
        """Gate ECG peaks and update the session's EMG signal and envelope.

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

        ecg_gating(
            session,
            processed,
            peak_indices,
            **self._kwargs_for(config, "ecg_gating"),
        )
