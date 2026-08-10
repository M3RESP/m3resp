"""Load/preprocess/detect orchestration and Layer 1 conversions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.signals import ProcessingState
from m3resp.synchronization.ventilator import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)

from ._channels import CHANNEL_CATEGORIES
from ._eit_source import DEFAULT_EIT_CHANNELS, ventilator_payload_from_sequence
from ._protocols import _DefaultsProtocol

#: File suffixes whose ventilator waveforms live inside an EIT recording.
_EIT_SUFFIXES = (".bin",)


def resolve_ventilator_source(path: Any, source: str | None = None) -> str:
    """Which modality's file a ventilator recording arrives in.

    ``"eit"`` or ``"emg"`` when the ventilator waveforms are carried inside
    that modality's own file, which means they share its clock: aligning that
    modality aligns the ventilator channels with it, and they must not also be
    shifted by a ventilator offset of their own. ``"emg"`` is the default
    because the multi-channel export shared with the sEMG is the common case.

    A standalone ventilator or monitor export has no such host and is aligned
    on its own, which is what `M3Session.synchronize_raw_modalities`'
    ventilator offset is for.
    """

    if source is None:
        return "eit" if str(path).lower().endswith(_EIT_SUFFIXES) else "emg"
    if source not in {"eit", "emg", "ventilator"}:
        raise ValueError(
            f"Ventilator load `source` must be 'eit', 'emg' or 'ventilator', "
            f"got {source!r}."
        )
    return source


class _CoreMixin:
    def __init__(
        self,
        loader: Callable[..., Any] | None = None,
        *,
        eit_loader: Callable[..., Any] | None = None,
    ):
        self._loader = loader
        self._eit_loader = eit_loader

    def load(self, path: str, **kwargs: Any) -> Any:
        """Load a ventilator recording from either of its two sources.

        Ventilator data reaches m3resp two ways, and which one applies is a
        property of the file, not of the caller:

        * a multi-channel file shared with the sEMG (Biopac exports and
          friends), read by
          :class:`~m3resp.adapters.resurfemg_adapter.ReSurfEMGAdapter`;
        * an EIT ``*.bin``, where the device stores ventilator waveforms
          beside the impedance frames (Draeger Medibus fields, Timpel columns),
          read through :class:`~m3resp.adapters.eitprocessing_adapter.EITProcessingAdapter`
          and unpacked by :mod:`._eit_source`.

        Dispatch is by suffix. Pass ``source="eit"`` or ``source="emg"`` to
        force one - useful for a file whose extension does not match its
        contents. ``ventilator_channels=`` selects which channels to pull from
        an EIT recording (default pressure/flow/volume; a Draeger pressure pod
        additionally offers esophageal, transpulmonary, and gastric pressure).

        Either source can be replaced with an injected callable: ``loader=``
        for the sEMG-file path, ``eit_loader=`` for the EIT one. Both return
        the same ``{"array", "metadata"}`` payload, so nothing downstream of
        loading needs to know which source a recording came from - except
        synchronization, which does: waveforms read out of the EIT or sEMG file
        share that modality's clock. See :func:`resolve_ventilator_source`.

        ``source="ventilator"`` marks a standalone ventilator/monitor export
        that happens to be readable by the sEMG loader but carries its own
        clock, so it is aligned on its own.
        """

        source = resolve_ventilator_source(path, kwargs.pop("source", None))

        if source == "eit":
            return self._load_from_eit(path, **kwargs)

        if self._loader is not None:
            return self._loader(path, **kwargs)

        from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter

        return ReSurfEMGAdapter().load(path, **kwargs)

    def _load_from_eit(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Load ventilator channels out of an EIT recording."""

        channels = kwargs.pop("ventilator_channels", DEFAULT_EIT_CHANNELS)
        fs = kwargs.pop("fs", None)

        if self._eit_loader is not None:
            sequence = self._eit_loader(path, **kwargs)
        else:
            from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter

            sequence = EITProcessingAdapter().load(str(path), **kwargs)

        return ventilator_payload_from_sequence(sequence, channels=channels, fs=fs)

    def preprocess(self, recording: Any, **kwargs: Any) -> Any:
        """Split into channels and filter, or defer to a provided callable."""

        preprocess = kwargs.pop("preprocess", None)
        if preprocess is not None:
            return preprocess(recording, **kwargs)

        return cast(_DefaultsProtocol, self)._preprocess_default(recording, **kwargs)

    def detect_breaths(self, processed: Any, **kwargs: Any) -> list[BreathEvent]:
        """Detect ventilator breaths and normalize them into `BreathEvent`s."""

        detector = kwargs.pop("detector", None)
        breath_width_seconds = kwargs.pop("breath_width_seconds", 0.5)
        if detector is not None:
            detections = detector(processed, **kwargs)
            sample_frequency = _sample_frequency(processed)
        else:
            detections = cast(_DefaultsProtocol, self)._detect_breaths_default(
                processed, breath_width_seconds=breath_width_seconds, **kwargs
            )
            sample_frequency = _sample_frequency(processed)

        return [
            normalize_ventilator_breath(
                detection,
                fs=sample_frequency,
                width_seconds=breath_width_seconds,
            )
            for detection in iter_ventilator_detections(detections)
        ]

    def to_signals(self, processed_ventilator: Any) -> list[Signal]:
        """Convert a preprocessed ventilator bundle into `Signal` objects.

        One signal per resolved channel per processing state: the unfiltered
        channel as ``"raw"`` and the filtered one as ``"processed"``. Each
        carries ``modality="ventilator"``, the channel's physical quantity in
        ``category``, its unique key in ``channel``, and the instrument it came
        from in ``source``.

        Those three axes are what let one session hold several pressures at
        once, including two airway pressures measured by different devices:
        they share a ``category`` and differ in ``channel``/``source``, so
        `SignalCollection.for_category("airway_pressure")` returns both instead
        of one overwriting the other.

        Units are passed through exactly as the vendor reported them and are
        never converted, so comparing two channels of one quantity has to
        account for their units (Draeger reports volume in mL, Timpel in L).
        """

        if (
            not isinstance(processed_ventilator, dict)
            or "fs" not in processed_ventilator
        ):
            raise UnsupportedWorkflowError(
                "to_signals expects the bundle from preprocess_ventilator()."
            )

        sample_frequency = float(processed_ventilator["fs"])
        units = processed_ventilator.get("units") or {}
        raw = processed_ventilator.get("raw") or {}
        filtered = processed_ventilator.get("filtered") or {}
        cutoff = (processed_ventilator.get("filter") or {}).get("lowpass_hz")
        method = (
            f"m3resp.processing.filters.lowpass_filter(cutoff_frequency={cutoff})"
            if cutoff is not None
            else None
        )

        specs = processed_ventilator.get("specs") or {}
        # Every channel the recording actually yielded, not a fixed three, so
        # an esophageal or a second airway pressure reaches `session.signals`.
        keys = list(processed_ventilator.get("channels") or raw or filtered)

        signals: list[Signal] = []
        for key in keys:
            spec = specs.get(key)
            category = (
                spec.category
                if spec is not None
                else CHANNEL_CATEGORIES.get(key.split("__", 1)[0])
            )
            channel_sources: tuple[tuple[Any, ProcessingState], ...] = (
                (raw.get(key), "raw"),
                (filtered.get(key), "processed"),
            )
            for values, processing_state in channel_sources:
                if values is None:
                    continue
                values = np.asarray(values, dtype=float)
                signals.append(
                    Signal(
                        values=values,
                        time=np.arange(values.shape[0], dtype=float) / sample_frequency,
                        sample_frequency=sample_frequency,
                        unit=units.get(key),
                        name=f"ventilator_{key}",
                        modality="ventilator",
                        category=category,
                        channel=key,
                        source=(spec.origin if spec is not None else None) or "m3resp",
                        processing_state=processing_state,
                        method=method if processing_state == "processed" else None,
                    )
                )
        return signals

    def to_parameters(self, processed_ventilator: Any) -> list[ParameterResult]:
        """No parameters are computed during ventilator preprocessing.

        Ventilator-derived metrics (Pocc time products, breath timing) come from
        the dedicated detection/quality steps, not from preprocessing. Present
        so the adapter surface matches the EIT and EMG adapters.
        """

        return []

    def to_quality_flags(self, processed_ventilator: Any) -> list[QualityFlag]:
        """No quality checks run during ventilator preprocessing (see
        `to_parameters`)."""

        return []


def _sample_frequency(processed: Any) -> float | None:
    if isinstance(processed, dict) and processed.get("fs") is not None:
        return float(processed["fs"])
    return None
