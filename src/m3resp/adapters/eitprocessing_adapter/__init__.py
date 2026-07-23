"""Adapter boundary for the upstream `eitprocessing` package.

This package mirrors the former single ``eitprocessing_adapter.py`` module,
with standalone helper functions factored out into ``_shared.py`` for
readability; ``EITProcessingAdapter`` itself is unchanged. ``add_to_collection``,
``continuous_data_to_signal``, and ``_sparse_data_to_parameters`` are
re-exported here so ``from m3resp.adapters.eitprocessing_adapter import
<name>`` keeps working unchanged.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.events import coerce_breath_events
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.processing.filters import butterworth_filter

from ._shared import (
    _breath_intervals_to_dicts,
    _lazy_import,
    _require_eit_sequence,
    _sparse_data_to_parameters,
    add_to_collection,
    continuous_data_to_signal,
)

__all__ = [
    "EITProcessingAdapter",
    "_sparse_data_to_parameters",
    "add_to_collection",
    "continuous_data_to_signal",
]


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

        if vendor is None:
            raise ValueError(
                "EIT loading requires an explicit vendor (e.g. 'draeger', "
                "'sentec', or 'timpel') when no loader is injected."
            )

        (load_eit_data,) = _lazy_import(
            "eitprocessing.datahandling.loading.load_eit_data",
            error=lambda: OptionalDependencyError(
                "EIT support requires the optional dependency `eitprocessing`. "
                'Install with `pip install "m3resp[eit]"` or inject a loader.'
            ),
        )

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
        add_to_collection(sequence.continuous_data, global_impedance)
        return global_impedance

    # -- Phase 1: reusable adapter operations ----------------------------------
    #
    # Small public methods, each wrapping exactly one `eitprocessing` operation.
    # `preprocess()` and the granular workflow steps in
    # `m3resp.workflows.steps.eit` both call these so the two paths cannot
    # drift. Upstream imports stay local to each method so `m3resp` installs
    # without the optional `eitprocessing` dependency.

    def detect_rates(
        self,
        signal: Any,
        *,
        subject_type: Literal["adult", "neonate"] = "adult",
        welch_window_seconds: float | None = None,
        capture: bool = False,
    ) -> dict[str, Any]:
        """Estimate respiratory and heart rate from an EIT (or continuous) signal."""

        (RateDetection,) = _lazy_import(
            "eitprocessing.features.rate_detection.RateDetection"
        )

        detector = (
            RateDetection(subject_type)
            if welch_window_seconds is None
            else RateDetection(subject_type, welch_window=welch_window_seconds)
        )
        captures: dict[str, Any] = {}
        if capture:
            respiratory_rate_hz, heart_rate_hz = detector.apply(
                signal,
                captures=captures,
                suppress_length_warnings=True,
                suppress_edge_case_warning=True,
            )
        else:
            respiratory_rate_hz, heart_rate_hz = detector.apply(signal)
        return {
            "respiratory_rate_hz": float(respiratory_rate_hz),
            "heart_rate_hz": float(heart_rate_hz),
            "rate_detector": detector,
            "rate_captures": captures,
        }

    def apply_mdn(
        self,
        signal: Any,
        *,
        respiratory_rate_hz: float,
        heart_rate_hz: float,
        label: str = "filtered",
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Apply an MDN heart-rate-removal filter to EIT pixel data."""

        (MDNFilter,) = _lazy_import("eitprocessing.filters.mdn.MDNFilter")

        for rate_name, rate in (
            ("respiratory_rate_hz", respiratory_rate_hz),
            ("heart_rate_hz", heart_rate_hz),
        ):
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(
                    f"{rate_name} must be a finite, positive value in Hz, got {rate!r}."
                )

        apply_kwargs: dict[str, Any] = {"label": label}
        if name is not None:
            apply_kwargs["name"] = name
        if description is not None:
            apply_kwargs["description"] = description

        captures: dict[str, Any] = {}
        filtered_eit = MDNFilter(
            respiratory_rate=respiratory_rate_hz,
            heart_rate=heart_rate_hz,
        ).apply(signal, captures=captures, **apply_kwargs)
        return {"filtered_eit": filtered_eit, "filter_captures": captures}

    def find_pixel_breaths(
        self,
        eit_data: Any,
        timing_data: Any,
        *,
        sequence: Any | None = None,
        phase_correction_mode: Literal["negative amplitude", "phase shift", "none"]
        | None = "negative amplitude",
        minimum_duration_seconds: float = 2 / 3,
        result_label: str = "pixel_breaths",
    ) -> Any:
        """Detect per-pixel breath timing (start/middle/end of in-/deflation)."""

        BreathDetection, PixelBreath = _lazy_import(
            "eitprocessing.features.breath_detection.BreathDetection",
            "eitprocessing.features.pixel_breath.PixelBreath",
        )

        breath_detector = BreathDetection(minimum_duration=minimum_duration_seconds)
        result = PixelBreath(
            breath_detection=breath_detector,
            phase_correction_mode=phase_correction_mode,
        ).find_pixel_breaths(
            eit_data,
            timing_data,
            sequence=sequence,
            store=False,
            result_label=result_label,
        )
        if sequence is not None:
            add_to_collection(sequence.interval_data, result)
        return result

    def compute_eeli(
        self,
        timing_data: Any,
        *,
        sequence: Any,
        breath_detector: Any,
        result_label: str = "continuous_eelis",
    ) -> Any:
        """Compute end-expiratory lung impedance (EELI) per breath."""

        (EELI,) = _lazy_import("eitprocessing.parameters.eeli.EELI")

        result = EELI(breath_detection=breath_detector).compute_parameter(
            timing_data,
            sequence=sequence,
            store=False,
            result_label=result_label,
        )
        add_to_collection(sequence.sparse_data, result)
        return result

    def compute_pixel_tiv(
        self,
        eit_data: Any,
        timing_data: Any,
        *,
        sequence: Any,
        breath_detector: Any,
        tiv_timing: Literal["pixel", "continuous"] = "continuous",
        result_label: str = "pixel_tivs",
    ) -> Any:
        """Compute per-pixel tidal impedance variation (TIV)."""

        (TIV,) = _lazy_import("eitprocessing.parameters.tidal_impedance_variation.TIV")

        result: Any = TIV(breath_detection=breath_detector).compute_parameter(
            eit_data,
            timing_data,
            sequence,
            tiv_timing=tiv_timing,
            store=False,
            result_label=result_label,
        )
        add_to_collection(sequence.sparse_data, result)
        return result

    def compute_tiv_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        threshold: float = 0.15,
    ) -> dict[str, Any]:
        """Threshold mean pixel TIV into a functional lung-space mask."""

        (TIVLungspace,) = _lazy_import("eitprocessing.roi.tiv.TIVLungspace")

        captures: dict[str, Any] = {}
        mask = TIVLungspace(threshold=threshold).apply(
            eit_data, timing_data=timing_data, captures=captures
        )
        return {"mask": mask, "captures": captures}

    def compute_amplitude_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        threshold: float = 0.15,
    ) -> dict[str, Any]:
        """Threshold mean pixel amplitude into a lung-space mask.

        Upstream does not recommend amplitude alone as a general-purpose
        functional lung-space definition; it exists primarily to support
        `compute_watershed_lungspace()`.
        """

        (AmplitudeLungspace,) = _lazy_import(
            "eitprocessing.roi.amplitude.AmplitudeLungspace"
        )

        captures: dict[str, Any] = {}
        mask = AmplitudeLungspace(threshold=threshold).apply(
            eit_data, timing_data=timing_data, captures=captures
        )
        return {"mask": mask, "captures": captures}

    def compute_watershed_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        threshold_fraction: float = 0.15,
    ) -> dict[str, Any]:
        """Derive a lung-space mask with the watershed method (pendelluft-aware)."""

        (WatershedLungspace,) = _lazy_import(
            "eitprocessing.roi.watershed.WatershedLungspace"
        )

        captures: dict[str, Any] = {}
        mask = WatershedLungspace(threshold_fraction=threshold_fraction).apply(
            eit_data, timing_data=timing_data, captures=captures
        )
        return {"mask": mask, "captures": captures}

    def filter_roi_by_size(
        self,
        mask: Any,
        *,
        min_region_size: int = 10,
        connectivity: Literal[1, 2] | np.ndarray = 1,
    ) -> Any:
        """Keep only connected mask regions at or above `min_region_size`."""

        (FilterROIBySize,) = _lazy_import(
            "eitprocessing.roi.filter_by_size.FilterROIBySize"
        )

        return FilterROIBySize(
            min_region_size=min_region_size, connectivity=connectivity
        ).apply(mask)

    def preprocess(
        self,
        sequence: Any,
        *,
        subject_type: Literal["adult", "neonate"] = "adult",
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

        BreathDetection, TIV = _lazy_import(
            "eitprocessing.features.breath_detection.BreathDetection",
            "eitprocessing.parameters.tidal_impedance_variation.TIV",
            error=lambda: OptionalDependencyError(
                "EIT preprocessing requires the optional dependency "
                '`eitprocessing`. Install with `pip install "m3resp[eit]"`.'
            ),
        )

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
            rates = self.detect_rates(
                raw_eit,
                subject_type=subject_type,
                welch_window_seconds=welch_window_seconds,
                capture=True,
            )
            rate_detector = rates["rate_detector"]
            rate_captures = rates["rate_captures"]
            respiratory_rate_hz = rates["respiratory_rate_hz"]
            heart_rate_hz = rates["heart_rate_hz"]

        filter_captures: dict[str, Any] = {}
        filtered_eit = raw_eit
        filtered_global_impedance = raw_global_impedance

        if normalized_filter_mode == "mdn":
            assert respiratory_rate_hz is not None
            assert heart_rate_hz is not None
            mdn_result = self.apply_mdn(
                raw_eit,
                respiratory_rate_hz=respiratory_rate_hz,
                heart_rate_hz=heart_rate_hz,
                label="mdn_filtered",
                name="MDN-filtered EIT data",
                description="EIT data filtered with MDN heart-rate noise removal.",
            )
            filtered_eit = mdn_result["filtered_eit"]
            filter_captures = mdn_result["filter_captures"]
        elif normalized_filter_mode in {"lowpass", "bandpass"}:
            butterworth_filter_type = cast(
                Literal["lowpass", "bandpass"], normalized_filter_mode
            )
            cutoff_frequency = (
                lowpass_hz
                if butterworth_filter_type == "lowpass"
                else (highpass_hz, lowpass_hz)
            )
            filtered_pixels = butterworth_filter(
                np.nan_to_num(raw_eit.pixel_impedance),
                filter_type=butterworth_filter_type,
                cutoff_frequency=cutoff_frequency,
                sample_frequency=raw_eit.sample_frequency,
                order=filter_order,
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
            add_to_collection(sequence.eit_data, filtered_eit)
            if include_global_impedance:
                filtered_global_impedance = filtered_eit.get_summed_impedance(
                    return_label=f"global_impedance_({filtered_eit.label})",
                    name=f"Global impedance ({filtered_eit.label})",
                    description="Global impedance calculated from filtered EIT data.",
                )
                add_to_collection(sequence.continuous_data, filtered_global_impedance)

        breath_detector = None
        breath_intervals = None
        continuous_tiv: Any = None
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
            add_to_collection(sequence.interval_data, breath_intervals)

        # `compute_breath_intervals` is guaranteed True here (validated above
        # whenever any of these three outputs is requested), so
        # `breath_detector`/`filtered_global_impedance` are always populated;
        # the asserts only narrow the type for mypy.
        if compute_continuous_tiv:
            assert breath_detector is not None
            assert filtered_global_impedance is not None
            tiv_calculator = TIV(breath_detection=breath_detector)
            continuous_tiv = tiv_calculator.compute_parameter(
                filtered_global_impedance,
                sequence=sequence,
                store=False,
                result_label="continuous_tivs",
            )
            add_to_collection(sequence.sparse_data, continuous_tiv)

        if compute_eeli:
            assert breath_detector is not None
            assert filtered_global_impedance is not None
            eeli = self.compute_eeli(
                filtered_global_impedance,
                sequence=sequence,
                breath_detector=breath_detector,
                result_label="continuous_eelis",
            )

        pixel_tiv: Any = None
        if compute_pixel_tiv:
            assert breath_detector is not None
            assert filtered_global_impedance is not None
            pixel_tiv = self.compute_pixel_tiv(
                filtered_eit,
                filtered_global_impedance,
                sequence=sequence,
                breath_detector=breath_detector,
                tiv_timing="continuous",
                result_label="pixel_tivs",
            )

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
                continuous_data_to_signal(
                    raw_gi,
                    modality="eit",
                    channel="global_impedance",
                    processing_state="raw",
                )
            )

        filtered_gi = preprocessed.get("filtered_global_impedance")
        if filtered_gi is not None and filtered_gi is not raw_gi:
            signals.append(
                continuous_data_to_signal(
                    filtered_gi,
                    modality="eit",
                    channel="global_impedance",
                    processing_state="filtered",
                    method=preprocessed.get("filter_mode"),
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
