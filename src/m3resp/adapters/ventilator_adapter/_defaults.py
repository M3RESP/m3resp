"""Ventilator channel extraction and breath detection defaults.

Preprocessing extracts named channels with recorded units. Supplying lowpass_hz
applies a Butterworth low-pass filter to each selected channel. Breath detection
uses the volume channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.processing.filters import lowpass_filter
from m3resp.processing.peaks import detect_ventilator_breath_peaks

from ._channels import DEFAULT_CHANNELS, primary_channel, split_channels

#: Suggested low-pass cutoff in Hz, applied when passed as lowpass_hz.
#: Choose a cutoff appropriate for the waveform features being studied.
SUGGESTED_LOWPASS_HZ = 20.0

#: Butterworth order used when a low-pass cutoff is requested.
DEFAULT_FILTER_ORDER = 4


class _DefaultsMixin:
    def _preprocess_default(
        self,
        recording: Any,
        *,
        channels: Any = DEFAULT_CHANNELS,
        airway_pressure_channel: int | None = None,
        flow_channel: int | None = None,
        volume_channel: int | None = None,
        channel_indices: dict[str, int] | None = None,
        origin: str | None = None,
        qualify: bool = False,
        fs: float | None = None,
        lowpass_hz: float | None = None,
        filter_order: int = DEFAULT_FILTER_ORDER,
    ) -> dict[str, Any]:
        """Extract selected ventilator channels and optionally low-pass filter them.

        Channel selection, explicit zero-based indices, origin, qualification and fs
        in Hz follow `split_channels`. The default selection is airway_pressure, flow
        and volume. All selected channels and their recorded units are retained.

        Args:
            recording (Any): Recording or (channels, samples) array accepted by
                `split_channels`.
            channels (Any): Requested channel names.
            airway_pressure_channel (int | None): Explicit airway-pressure index.
            flow_channel (int | None): Explicit flow index.
            volume_channel (int | None): Explicit volume index.
            channel_indices (dict[str, int] | None): Indices for requested channels.
            origin (str | None): Instrument or file name for channel metadata.
            qualify (bool): Add origin to channel keys when supplied.
            fs (float | None): Sampling rate in Hz, overriding metadata fs.
            lowpass_hz (float | None): Low-pass cutoff in Hz. None returns copied raw
                values. A supplied cutoff is limited to 95% of the Nyquist frequency
                (half the sampling rate).
            filter_order (int): Positive Butterworth order. Defaults to 4.

        Returns:
            dict[str, Any]: Channel bundle with original arrays under raw and copies
                or filtered arrays at each channel key and under channels. With a
                cutoff, filtered contains those filtered arrays; otherwise it is empty.
                The filter mapping records the requested and applied cutoffs and order.

        Raises:
            UnresolvedChannelError: If a selected channel cannot be found.
            ValueError: If a requested filter has invalid parameters or input samples.
        """

        bundle = split_channels(
            recording,
            channels=channels,
            airway_pressure_channel=airway_pressure_channel,
            flow_channel=flow_channel,
            volume_channel=volume_channel,
            channel_indices=channel_indices,
            origin=origin,
            qualify=qualify,
            fs=fs,
        )
        sample_frequency = float(bundle["fs"])
        raw = dict(bundle["channels"])

        cutoff = _resolve_cutoff(lowpass_hz, sample_frequency)
        if cutoff is None:
            # Nothing was filtered, so there is no processed version to offer:
            # `filtered` stays empty and every channel is emitted once, as raw.
            channels = {name: values.copy() for name, values in raw.items()}
            filtered: dict[str, Any] = {}
        else:
            channels = {
                name: lowpass_filter(
                    values,
                    cutoff_frequency=cutoff,
                    sample_frequency=sample_frequency,
                    order=filter_order,
                )
                for name, values in raw.items()
            }
            filtered = channels

        return {
            **bundle,
            **channels,
            # `channels` tracks the top-level keys, which expose the filtered
            # arrays when a cutoff was applied; the unfiltered ones are always
            # reachable under `raw`.
            "channels": channels,
            "raw": raw,
            "filtered": filtered,
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
        channel: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Detect ventilator breath peaks on the volume channel.

        Which channel that is comes from the bundle's ``primary`` map, so a
        recording carrying more than one volume trace detects on a defined one
        rather than whichever happened to be stored last. Pass ``channel=`` to
        detect on a specific one.
        """

        if not isinstance(processed_ventilator, dict):
            raise UnsupportedWorkflowError(
                "Default ventilator breath detection expects the bundle from "
                "`preprocess_ventilator()`. Pass `detector=callable` to "
                "normalize custom detections."
            )

        key = channel or primary_channel(processed_ventilator, "volume")
        if key is None or key not in processed_ventilator:
            raise UnsupportedWorkflowError(
                "Default ventilator breath detection needs a volume channel; "
                f"this bundle has {sorted(processed_ventilator.get('channels', {}))}. "
                "Pass `channel=` to detect on a different one, or "
                "`detector=callable` to normalize custom detections."
            )

        volume = np.asarray(processed_ventilator[key], dtype=float)
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
