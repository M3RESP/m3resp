"""Adapter boundary for the upstream `eitprocessing` package."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.events import coerce_breath_events
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.signals import Modality, ProcessingState


class EITProcessingAdapter:
    """Thin wrapper around `eitprocessing`.

    Stage 1 keeps this adapter deliberately small. It imports `eitprocessing`
    only when used so `m3resp` can be installed without optional EIT support.
    """

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, path: str, vendor: str | None = None, **kwargs: Any) -> Any:
        """Load EIT data through `eitprocessing` or an injected loader."""

        if self._loader is not None:
            return self._loader(path, vendor=vendor, **kwargs)

        try:
            from eitprocessing.datahandling.loading import load_eit_data
        except ImportError as exc:
            raise OptionalDependencyError(
                "EIT support requires the optional dependency `eitprocessing`. "
                'Install with `pip install "m3resp[eit]"` or inject a loader.'
            ) from exc

        return load_eit_data(path, vendor=vendor, **kwargs)

    def get_raw_eit(self, sequence: Any, label: str = "raw") -> Any:
        """Return the raw EIT data object from an upstream sequence."""

        return sequence.eit_data[label]

    def get_global_impedance(self, sequence: Any, label: str = "raw") -> Any:
        """Return and store global impedance from a loaded EIT sequence."""

        continuous_label = f"global_impedance_({label})"
        if continuous_label in sequence.continuous_data:
            return sequence.continuous_data[continuous_label]

        eit_data = self.get_raw_eit(sequence, label=label)
        global_impedance = eit_data.get_summed_impedance(
            return_label=continuous_label,
            name=f"Global impedance ({label})",
            description="Global impedance calculated from EIT pixel data.",
        )
        _add_to_collection(sequence.continuous_data, global_impedance)
        return global_impedance

    def preprocess(
        self,
        sequence: Any,
        *,
        subject_type: str = "adult",
        welch_window_seconds: float = 30.0,
        filter_mode: str = "mdn",
        filter_enabled: bool = True,
        lowpass_hz: float = 1.0,
        highpass_hz: float = 0.05,
        filter_order: int = 4,
        breath_min_duration_seconds: float = 2 / 3,
        compute_rates: bool = True,
        compute_breath_intervals: bool = True,
        compute_continuous_tiv: bool = True,
        compute_eeli: bool = True,
        compute_pixel_tiv: bool = True,
        include_filtered_data: bool = True,
        include_global_impedance: bool = True,
    ) -> dict[str, Any]:
        """Run the Stage 1 EIT preprocessing pipeline through `eitprocessing`."""

        try:
            import numpy as np
            from eitprocessing.features.breath_detection import BreathDetection
            from eitprocessing.features.rate_detection import RateDetection
            from eitprocessing.filters.butterworth_filters import ButterworthFilter
            from eitprocessing.filters.mdn import MDNFilter
            from eitprocessing.parameters.eeli import EELI
            from eitprocessing.parameters.tidal_impedance_variation import TIV
        except ImportError as exc:
            raise OptionalDependencyError(
                "EIT preprocessing requires the optional dependency "
                '`eitprocessing`. Install with `pip install "m3resp[eit]"`.'
            ) from exc

        _require_eit_sequence(sequence)

        if not compute_breath_intervals and (
            compute_continuous_tiv or compute_eeli or compute_pixel_tiv
        ):
            raise ValueError(
                "EIT TIV, EELI, and pixel TIV require breath intervals. "
                "Enable eit.processing.outputs.breath_intervals or disable "
                "the dependent EIT outputs."
            )

        raw_eit = self.get_raw_eit(sequence)
        normalized_filter_mode = "none" if not filter_enabled else filter_mode.lower()
        if normalized_filter_mode not in {"mdn", "lowpass", "bandpass", "none"}:
            raise ValueError(
                "filter_mode must be one of: 'mdn', 'lowpass', 'bandpass', 'none'."
            )

        raw_global_impedance = (
            self.get_global_impedance(sequence) if include_global_impedance else None
        )

        rate_detector = None
        rate_captures: dict[str, Any] = {}
        respiratory_rate_hz = None
        heart_rate_hz = None
        rates_required = compute_rates or normalized_filter_mode == "mdn"
        if rates_required:
            rate_detector = RateDetection(
                subject_type, welch_window=welch_window_seconds
            )
            respiratory_rate_hz, heart_rate_hz = rate_detector.apply(
                raw_eit,
                captures=rate_captures,
                suppress_length_warnings=True,
                suppress_edge_case_warning=True,
            )

        filter_captures: dict[str, Any] = {}
        filtered_eit = raw_eit
        filtered_global_impedance = raw_global_impedance

        if normalized_filter_mode == "mdn":
            eit_filter = MDNFilter(
                respiratory_rate=respiratory_rate_hz,
                heart_rate=heart_rate_hz,
            )
            filtered_eit = eit_filter.apply(
                raw_eit,
                captures=filter_captures,
                label="mdn_filtered",
                name="MDN-filtered EIT data",
                description="EIT data filtered with MDN heart-rate noise removal.",
            )
        elif normalized_filter_mode in {"lowpass", "bandpass"}:
            cutoff_frequency = (
                lowpass_hz
                if normalized_filter_mode == "lowpass"
                else (highpass_hz, lowpass_hz)
            )
            eit_filter = ButterworthFilter(
                filter_type=normalized_filter_mode,
                cutoff_frequency=cutoff_frequency,
                order=filter_order,
                sample_frequency=raw_eit.sample_frequency,
            )
            filtered_pixels = eit_filter.apply(
                np.nan_to_num(raw_eit.pixel_impedance),
                axis=0,
                captures=filter_captures,
            )
            filtered_eit = copy.deepcopy(raw_eit)
            filtered_eit.label = f"{normalized_filter_mode}_filtered"
            filtered_eit.name = f"{normalized_filter_mode.title()}-filtered EIT data"
            filtered_eit.description = (
                f"EIT data filtered with a {normalized_filter_mode} Butterworth filter."
            )
            filtered_eit.pixel_impedance = filtered_pixels

        if filtered_eit is not raw_eit:
            _add_to_collection(sequence.eit_data, filtered_eit)
            if include_global_impedance:
                filtered_global_impedance = filtered_eit.get_summed_impedance(
                    return_label=f"global_impedance_({filtered_eit.label})",
                    name=f"Global impedance ({filtered_eit.label})",
                    description="Global impedance calculated from filtered EIT data.",
                )
                _add_to_collection(sequence.continuous_data, filtered_global_impedance)

        breath_detector = None
        breath_intervals = None
        continuous_tiv = None
        eeli = None
        if compute_breath_intervals:
            if filtered_global_impedance is None:
                filtered_global_impedance = self.get_global_impedance(
                    sequence, label=filtered_eit.label
                )
            breath_detector = BreathDetection(
                minimum_duration=breath_min_duration_seconds
            )
            breath_intervals = breath_detector.find_breaths(
                filtered_global_impedance,
                result_label="eit_breaths",
                store=False,
            )
            _add_to_collection(sequence.interval_data, breath_intervals)

        if compute_continuous_tiv:
            tiv_calculator = TIV(breath_detection=breath_detector)
            continuous_tiv = tiv_calculator.compute_parameter(
                filtered_global_impedance,
                sequence=sequence,
                store=False,
                result_label="continuous_tivs",
            )
            _add_to_collection(sequence.sparse_data, continuous_tiv)

        if compute_eeli:
            eeli = EELI(breath_detection=breath_detector).compute_parameter(
                filtered_global_impedance,
                sequence=sequence,
                store=False,
                result_label="continuous_eelis",
            )
            _add_to_collection(sequence.sparse_data, eeli)

        pixel_tiv = None
        if compute_pixel_tiv:
            tiv_calculator = TIV(breath_detection=breath_detector)
            pixel_tiv = tiv_calculator.compute_parameter(
                filtered_eit,
                filtered_global_impedance,
                sequence,
                tiv_timing="continuous",
                store=False,
                result_label="pixel_tivs",
            )
            _add_to_collection(sequence.sparse_data, pixel_tiv)

        return {
            "sequence": sequence,
            "raw_eit": raw_eit,
            "raw_global_impedance": raw_global_impedance,
            "filtered_eit": filtered_eit if include_filtered_data else None,
            "filtered_global_impedance": filtered_global_impedance,
            "filter_mode": normalized_filter_mode,
            "filter_captures": filter_captures,
            "rate_detector": rate_detector,
            "rate_captures": rate_captures,
            "respiratory_rate_hz": respiratory_rate_hz,
            "heart_rate_hz": heart_rate_hz,
            "breath_intervals": breath_intervals,
            "continuous_tiv": continuous_tiv,
            "eeli": eeli,
            "pixel_tiv": pixel_tiv,
        }

    def detect_breaths(self, data: Any, **kwargs: Any) -> list[BreathEvent]:
        """Normalize upstream EIT breath detections into `BreathEvent` objects."""

        detector = kwargs.pop("detector", None)
        if detector is None:
            if not isinstance(data, dict) or "breath_intervals" not in data:
                raise UnsupportedWorkflowError(
                    "Default EIT breath detection expects processed EIT data from "
                    "`preprocess_eit()`. Pass `detector=callable` to normalize "
                    "custom detections."
                )
            detections = _breath_intervals_to_dicts(data["breath_intervals"])
            return coerce_breath_events(
                detections,
                modality="eit",
                source="eitprocessing.BreathDetection",
            )

        detections = detector(data, **kwargs)
        return coerce_breath_events(detections, modality="eit", source="eitprocessing")

    def compute_tiv(self, sequence: Any, **kwargs: Any) -> Any:
        """Compute tidal impedance variation when an upstream function is provided."""

        compute = kwargs.pop("compute", None)
        if compute is None:
            raise UnsupportedWorkflowError(
                "TIV computation needs an upstream callable in Stage 1. "
                "Pass `compute=callable`."
            )
        return compute(sequence, **kwargs)

    # -- Milestone 2.3: conversion to m3resp-native Layer 1 objects -----------
    #
    # These convert `preprocess()`'s output dict (still raw `eitprocessing`
    # `ContinuousData`/`SparseData` objects) into `m3resp.data` objects. They
    # duck-type on `.values`/`.time`/`.sample_frequency`/`.unit`/`.name` rather
    # than importing `eitprocessing` types, so they also work against any
    # object with that shape (e.g. in tests, without the optional dependency).

    def to_signals(self, preprocessed: dict[str, Any]) -> list[Signal]:
        """Convert preprocessed global-impedance waveforms into `Signal` objects."""

        signals: list[Signal] = []
        raw_gi = preprocessed.get("raw_global_impedance")
        if raw_gi is not None:
            signals.append(
                _continuous_data_to_signal(
                    raw_gi,
                    modality="eit",
                    channel="global_impedance",
                    processing_state="raw",
                )
            )

        filtered_gi = preprocessed.get("filtered_global_impedance")
        if filtered_gi is not None and filtered_gi is not raw_gi:
            signals.append(
                _continuous_data_to_signal(
                    filtered_gi,
                    modality="eit",
                    channel="global_impedance",
                    processing_state="filtered",
                )
            )
        return signals

    def to_parameters(self, preprocessed: dict[str, Any]) -> list[ParameterResult]:
        """Convert rate/TIV/EELI results into `ParameterResult` objects."""

        parameters: list[ParameterResult] = []

        respiratory_rate_hz = preprocessed.get("respiratory_rate_hz")
        if respiratory_rate_hz is not None:
            parameters.append(
                ParameterResult(
                    name="respiratory_rate",
                    value=float(respiratory_rate_hz),
                    modality="eit",
                    unit="Hz",
                    method="eitprocessing.RateDetection",
                )
            )

        heart_rate_hz = preprocessed.get("heart_rate_hz")
        if heart_rate_hz is not None:
            parameters.append(
                ParameterResult(
                    name="heart_rate",
                    value=float(heart_rate_hz),
                    modality="eit",
                    unit="Hz",
                    method="eitprocessing.RateDetection",
                )
            )

        for key, method in (
            ("continuous_tiv", "eitprocessing.TIV"),
            ("eeli", "eitprocessing.EELI"),
            ("pixel_tiv", "eitprocessing.TIV"),
        ):
            sparse = preprocessed.get(key)
            if sparse is not None:
                parameters.extend(
                    _sparse_data_to_parameters(sparse, modality="eit", method=method)
                )
        return parameters

    def to_quality_flags(self, preprocessed: dict[str, Any]) -> list[QualityFlag]:
        """Convert preprocessing completeness into `QualityFlag` objects.

        `eitprocessing` does not expose an explicit signal-quality score at
        this stage, so these flags report whether the optional preprocessing
        outputs a downstream step depends on (rate detection, breath
        intervals) were actually produced.
        """

        return [
            QualityFlag(
                name="respiratory_rate_detected",
                passed=preprocessed.get("respiratory_rate_hz") is not None,
                severity="warning",
                modality="eit",
            ),
            QualityFlag(
                name="breath_intervals_detected",
                passed=preprocessed.get("breath_intervals") is not None,
                severity="warning",
                modality="eit",
            ),
        ]


def _require_eit_sequence(sequence: Any) -> None:
    if not hasattr(sequence, "eit_data") or not hasattr(sequence, "continuous_data"):
        raise TypeError("EIT preprocessing expects an eitprocessing Sequence.")
    if "raw" not in sequence.eit_data:
        raise KeyError("EIT preprocessing requires sequence.eit_data['raw'].")


def _add_to_collection(collection: Any, value: Any) -> None:
    try:
        collection.add(value, overwrite=True)
    except TypeError:
        collection.add(value)


def _breath_intervals_to_dicts(breath_intervals: Any) -> list[dict[str, Any]]:
    return [
        {
            "start_time": breath.start_time,
            "end_time": breath.end_time,
            "peak_time": getattr(breath, "middle_time", None),
            "source": "eitprocessing.BreathDetection",
        }
        for breath in breath_intervals.values
    ]


def _continuous_data_to_signal(
    obj: Any,
    *,
    modality: Modality,
    channel: str | None,
    processing_state: ProcessingState,
) -> Signal:
    """Convert an `eitprocessing.ContinuousData`-shaped object to a `Signal`."""

    return Signal(
        values=obj.values,
        time=obj.time,
        sample_frequency=getattr(obj, "sample_frequency", None),
        unit=getattr(obj, "unit", None),
        name=getattr(obj, "name", None) or getattr(obj, "label", None),
        modality=modality,
        channel=channel,
        processing_state=processing_state,
    )


def _sparse_data_to_parameters(
    obj: Any, *, modality: str, method: str
) -> list[ParameterResult]:
    """Convert an `eitprocessing.SparseData`-shaped object (one value per
    breath) into one `ParameterResult` per non-NaN sample.
    """

    values = np.asarray(obj.values)
    times = np.asarray(obj.time)
    name = getattr(obj, "name", None) or getattr(obj, "label", None) or "parameter"
    unit = getattr(obj, "unit", None)

    results: list[ParameterResult] = []
    for index, value in enumerate(values):
        if np.ndim(value) == 0 and np.isnan(value):
            continue
        metadata = {"time": float(times[index])} if index < len(times) else {}
        results.append(
            ParameterResult(
                name=name,
                value=value,
                modality=modality,
                unit=unit,
                breath_id=str(index),
                method=method,
                metadata=metadata,
            )
        )
    return results
