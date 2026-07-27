"""Native ventilator preprocessing and breath detection defaults.

Unlike the EIT and EMG adapters, this one has no upstream library to wrap:
neither `eitprocessing` nor `resurfemg` implements ventilator preprocessing, so
there was nothing to borrow and ventilator channels went unfiltered. These
defaults are therefore native from the start, built on
:mod:`m3resp.processing.filters` and :mod:`m3resp.processing.peaks` - the
direction Stage 3 takes for every operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.processing.filters import lowpass_filter
from m3resp.processing.peaks import detect_ventilator_breath_peaks

from ._channels import split_channels

#: Default low-pass cutoff applied to each ventilator channel, in Hz.
#:
#: Deliberately conservative and *not* a clinical parameter: respiratory
#: waveform content sits below roughly 5 Hz, so a 20 Hz cutoff removes sensor
#: and quantization noise while leaving breath morphology (including the sharp
#: pressure upstroke that Pocc quality assessment measures) untouched. Raise or
#: lower it per recording via ``lowpass_hz``, or pass ``lowpass_hz=None`` to
#: skip filtering entirely.
DEFAULT_LOWPASS_HZ = 20.0

#: Butterworth order for the channel low-pass.
DEFAULT_FILTER_ORDER = 4

_CHANNEL_NAMES = ("pressure", "flow", "volume")


class _DefaultsMixin:
    def _preprocess_default(
        self,
        recording: Any,
        *,
        pressure_channel: int = 0,
        flow_channel: int = 1,
        volume_channel: int = 2,
        fs: float | None = None,
        lowpass_hz: float | None = DEFAULT_LOWPASS_HZ,
        filter_order: int = DEFAULT_FILTER_ORDER,
    ) -> dict[str, Any]:
        """Split a ventilator recording into channels and low-pass each one.

        The returned bundle keeps the unfiltered arrays under ``"raw"`` and
        exposes the filtered ones under the plain ``"pressure"``/``"flow"``/
        ``"volume"`` keys, so downstream consumers get the processed signal by
        default while the originals stay available - the same arrangement as
        the EMG bundle's ``raw_channel``/``filtered``/``envelope``.

        ``lowpass_hz=None`` skips filtering, in which case the filtered and raw
        arrays are the same values and ``processing_state`` stays ``"raw"``.
        """

        bundle = split_channels(
            recording,
            pressure_channel=pressure_channel,
            flow_channel=flow_channel,
            volume_channel=volume_channel,
            fs=fs,
        )
        sample_frequency = float(bundle["fs"])
        raw = {name: bundle[name] for name in _CHANNEL_NAMES}

        cutoff = _resolve_cutoff(lowpass_hz, sample_frequency)
        if cutoff is None:
            processed = {name: values.copy() for name, values in raw.items()}
        else:
            processed = {
                name: lowpass_filter(
                    values,
                    cutoff_frequency=cutoff,
                    sample_frequency=sample_frequency,
                    order=filter_order,
                )
                for name, values in raw.items()
            }

        return {
            **bundle,
            **processed,
            "raw": raw,
            "filtered": processed,
            "filter": {
                "requested_lowpass_hz": lowpass_hz,
                "lowpass_hz": cutoff,
                "filter_order": filter_order if cutoff is not None else None,
            },
        }

    def _detect_breaths_default(
        self,
        processed_ventilator: Any,
        *,
        breath_width_seconds: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Detect ventilator breath peaks on the volume channel."""

        if (
            not isinstance(processed_ventilator, dict)
            or "volume" not in processed_ventilator
        ):
            raise UnsupportedWorkflowError(
                "Default ventilator breath detection expects the bundle from "
                "`preprocess_ventilator()`. Pass `detector=callable` to "
                "normalize custom detections."
            )

        volume = np.asarray(processed_ventilator["volume"], dtype=float)
        sample_frequency = float(processed_ventilator["fs"])
        width_samples = max(1, int(breath_width_seconds * sample_frequency))
        return detect_ventilator_breath_peaks(
            volume,
            start_index=0,
            end_index=len(volume) - 1,
            width_samples=width_samples,
            **kwargs,
        )


def _resolve_cutoff(lowpass_hz: float | None, sample_frequency: float) -> float | None:
    """Clamp the requested cutoff below Nyquist, or return ``None`` to skip.

    A recording sampled slower than twice the requested cutoff would otherwise
    raise; clamping keeps a low-rate ventilator export usable instead.
    """

    if lowpass_hz is None:
        return None
    nyquist = sample_frequency / 2
    if nyquist <= 0:
        return None
    return float(min(float(lowpass_hz), nyquist * 0.95))
