"""`EMGPreset`: the built-in "emg" preset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.presets.base import Preset, PresetConfig

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


#: Settings the preset passes to `M3Session.emg_moving_baseline`.
_BASELINE_DEFAULTS = {"window_seconds": 7.5, "step_seconds": 0.2, "percentile": 33.0}

#: Config groups `EMGPreset` reads.
_CONFIG_GROUPS = frozenset({"preprocess", "baseline", "detect_breaths", "postprocess"})

#: Groups from before ECG removal and the envelope moved into preprocessing,
#: mapped to where their options go now.
_MOVED_GROUPS = {
    "ecg_removal": "'remove_ecg' and 'ecg_peak_indices'",
    "ecg_detect_peaks": "'ecg_channel'",
    "ecg_gating": "'ecg_removal_method' and 'ecg_gate_fill_method'",
    "envelope": "'compute_envelope', 'envelope_method' and 'envelope_window_seconds'",
}


class EMGPreset(Preset):
    """Preprocess loaded EMG, detect breaths and calculate features.

    Runs preprocessing, a moving baseline, breath detection and
    postprocessing in that order. The default preprocessing function
    band-passes the signal, detects ECG peaks, removes ECG and computes the
    envelope, so all of those options go under ``config["preprocess"]``
    (e.g. ``remove_ecg``, ``ecg_channel``, ``ecg_peak_indices``,
    ``ecg_removal_method``, ``ecg_gate_fill_method``, ``envelope_method``,
    ``envelope_window_seconds``).

    Config groups keyword arguments under ``preprocess``, ``baseline``,
    ``detect_breaths`` and ``postprocess``. Baseline options are
    ``window_seconds`` (7.5 s by default), ``step_seconds`` (0.2 s) and
    ``percentile`` (33); other keys raise TypeError. The baseline is skipped
    when preprocessing produced no envelope.

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
                operation defaults, including ECG removal and moving-baseline
                estimation before breath detection.

        Returns:
            M3Session: The supplied session with processed signals, detected
                breaths, parameters, quality flags and provenance.

        Raises:
            MissingModalityDataError: If the required EMG recording is unavailable.
            TypeError: If config contains unknown groups or baseline options.
        """
        self._check_config(config)
        session.preprocess_emg(**self._kwargs_for(config, "preprocess"))
        baseline = self._moving_baseline(session, config)
        session.detect_emg_breaths(
            baseline=baseline, **self._kwargs_for(config, "detect_breaths")
        )
        session.postprocess_emg(**self._kwargs_for(config, "postprocess"))
        return session

    @staticmethod
    def _check_config(config: PresetConfig | None) -> None:
        """Raise TypeError for config groups this preset does not read."""
        groups = set(config or {})
        moved = sorted(groups & set(_MOVED_GROUPS))
        if moved:
            hints = "; ".join(
                f"config['{group}'] -> {_MOVED_GROUPS[group]}" for group in moved
            )
            msg = (
                "EMGPreset: ECG removal and the envelope are now part of "
                f"preprocessing; pass these under config['preprocess']: {hints}."
            )
            raise TypeError(msg)
        unknown = sorted(groups - _CONFIG_GROUPS)
        if unknown:
            msg = f"EMGPreset config accepts {sorted(_CONFIG_GROUPS)}; got {unknown}."
            raise TypeError(msg)

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
            msg = (
                f"EMGPreset config['baseline'] accepts {sorted(_BASELINE_DEFAULTS)}; "
                f"got {unknown}. The baseline is computed from the same processed "
                "EMG that breath detection uses."
            )
            raise TypeError(msg)
        settings = {
            name: float(options.get(name, default))
            for name, default in _BASELINE_DEFAULTS.items()
        }
        return session.emg_moving_baseline(**settings)
