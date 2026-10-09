"""Load and process EIT recordings through eitprocessing.

Conversion helpers produce m3resp signals, breath events, rate parameters and
values per breath from eitprocessing results.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import IntervalData, ParameterResult, PixelMask, QualityFlag, Signal
from m3resp.data.events import (
    BreathEvent,
    coerce_breath_events,
    reuse_matching_breaths,
)
from m3resp.processing.filters import butterworth_filter

from ._shared import (
    _breath_intervals_to_dicts,
    _lazy_import,
    _require_eit_sequence,
    add_to_collection,
    breath_intervals_to_breath_events,
    continuous_data_to_signal,
    filter_pixels_preserving_gaps,
    sparse_data_to_interval_data,
)

__all__ = [
    "EITProcessingAdapter",
    "add_to_collection",
    "breath_intervals_to_breath_events",
    "continuous_data_to_signal",
    "filter_pixels_preserving_gaps",
    "sparse_data_to_interval_data",
]


class EITProcessingAdapter:
    """Thin wrapper around `eitprocessing`.

    It imports `eitprocessing` only when used, so `m3resp` can be installed
    without optional EIT support.
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

    def get_regional_impedance(self, eit_data: Any, mask: Any, *, label: str) -> Any:
        """Return the impedance summed over the pixels of a mask, per frame.

        Each pixel is multiplied by its mask value, so a NaN pixel drops out
        and a weighted pixel counts partly; the result is summed over all
        pixels in each frame. This is eitprocessing's ``PixelMask.apply``
        followed by ``EITData.get_summed_impedance``.

        Args:
            eit_data: eitprocessing pixel data with time, row and column axes.
            mask: An eitprocessing or m3resp ``PixelMask``, or a 2D grid with
                NaN for pixels outside the region.
            label: Label of the returned waveform, e.g.
                ``'functional_impedance'``.

        Returns:
            eitprocessing ``ContinuousData``: the regional impedance waveform
                on the time axis of ``eit_data``, in arbitrary units (AU).

        Raises:
            UnsupportedWorkflowError: If ``mask`` cannot be read as a 2D mask.
            ValueError: If the mask shape does not match the image shape.
        """

        upstream_mask = self.as_pixel_mask(mask)
        masked = upstream_mask.apply(eit_data)
        return masked.get_summed_impedance(
            return_label=label,
            name=label.replace("_", " ").capitalize(),
            description="Impedance summed over the pixels of a mask.",
        )

    def slice_sequence(self, sequence: Any, start_index: int, end_index: int) -> Any:
        """Return a copy of `sequence` keeping frames `start_index` up to (not
        including) `end_index`.

        Uses `eitprocessing`'s own `Sequence.select_by_index`, which cuts the
        pixel data, every continuous signal (global impedance, pressures,
        flow) and the vendor's markers to the same frames. The time values
        are kept as they were, so the first kept frame keeps its original
        time stamp.
        """

        select_by_index = getattr(sequence, "select_by_index", None)
        if select_by_index is None:
            raise TypeError(
                "Slicing EIT data needs an eitprocessing Sequence (or an "
                "object with select_by_index); got "
                f"{type(sequence).__name__}."
            )
        return select_by_index(start=start_index, end=end_index)

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

    def find_breaths(
        self, timing_data: Any, *, minimum_duration_seconds: float = 2 / 3
    ) -> Any:
        """Detect breaths on a global or regional impedance waveform.

        Args:
            timing_data: eitprocessing ``ContinuousData`` containing the
                waveform and its time axis in seconds.
            minimum_duration_seconds: Minimum separation used by the breath
                detector, in seconds.

        Returns:
            eitprocessing.IntervalData: Detected breaths with start, middle
                and end times on the waveform's time axis.

        Raises:
            OptionalDependencyError: If eitprocessing is unavailable.
        """

        (BreathDetection,) = _lazy_import(
            "eitprocessing.features.breath_detection.BreathDetection"
        )
        return BreathDetection(minimum_duration=minimum_duration_seconds).find_breaths(
            timing_data
        )

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
        """Keep mask regions containing at least the specified number of pixels.

        Args:
            mask: An eitprocessing or m3resp ``PixelMask``, or a 2D numeric
                grid with NaN for excluded pixels.
            min_region_size: Minimum number of connected pixels to retain.
            connectivity: ``1`` joins edge neighbours; ``2`` also joins
                diagonal neighbours. A custom connection array is accepted.

        Returns:
            eitprocessing.PixelMask: Mask with small connected regions removed.

        Raises:
            OptionalDependencyError: If eitprocessing is unavailable.
            UnsupportedWorkflowError: If the mask cannot be converted to a
                2D numeric grid.
        """

        (FilterROIBySize,) = _lazy_import(
            "eitprocessing.roi.filter_by_size.FilterROIBySize"
        )

        return FilterROIBySize(
            min_region_size=min_region_size, connectivity=connectivity
        ).apply(self.as_pixel_mask(mask))

    def as_pixel_mask(self, mask: Any) -> Any:
        """Convert a mask to eitprocessing's ``PixelMask`` representation.

        Args:
            mask: An eitprocessing ``PixelMask``, an m3resp ``PixelMask``, or
                a 2D numeric array, list or tuple. Numeric grids use NaN for
                excluded pixels and weights from 0 to 1 for included pixels.
                An m3resp ``PixelMask`` keeps its zeros as weight 0; in a
                plain grid, eitprocessing turns zeros into NaN.

        Returns:
            eitprocessing.PixelMask: An existing object with a ``mask``
                attribute is returned unchanged. Other inputs are converted
                to a float grid in row-column order.

        Raises:
            UnsupportedWorkflowError: If the input has an unsupported type
                or cannot be converted to a 2D numeric grid.
            OptionalDependencyError: If conversion requires eitprocessing
                and it is unavailable.
        """

        if hasattr(mask, "mask"):
            return mask

        value = mask.values if isinstance(mask, PixelMask) else mask
        if not isinstance(value, (np.ndarray, list, tuple)):
            raise UnsupportedWorkflowError(
                "An ROI mask must be an m3resp PixelMask, an eitprocessing "
                f"PixelMask, or a 2D array of pixels; got {type(mask).__name__}."
            )
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise UnsupportedWorkflowError(
                "An ROI mask must be a 2D (row, column) array of numbers."
            ) from error
        if array.ndim != 2:
            raise UnsupportedWorkflowError(
                "An ROI mask must be a 2D (row, column) array of pixels; got "
                f"shape {array.shape}."
            )

        (UpstreamPixelMask,) = _lazy_import("eitprocessing.roi.PixelMask")
        if isinstance(mask, PixelMask):
            # An m3resp mask has already been checked, so its zeros (weight 0)
            # and weights are passed on unchanged.
            return UpstreamPixelMask(
                array, keep_zeros=True, suppress_value_range_error=True
            )
        return UpstreamPixelMask(array)

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
        """Filter EIT data and compute selected breath-related results.

        Adds computed signals, breath intervals and measurements to the supplied
        sequence. MDN filtering requires respiratory and heart rates, so those
        rates are calculated even when compute_rates is False.

        Args:
            sequence: Loaded eitprocessing Sequence containing pixel impedance data.
            subject_type: adult or neonate, used for rate detection.
            welch_window_seconds: Rate-estimation window duration in seconds.
            filter_mode: mdn, lowpass, bandpass or none.
            filter_enabled: Use the selected filter when True; otherwise use none.
            lowpass_hz: Low-pass cutoff, or band-pass upper cutoff, in Hz.
            highpass_hz: Band-pass lower cutoff in Hz.
            filter_order: Butterworth filter order for lowpass or bandpass.
            breath_min_duration_seconds: Minimum detected breath duration in seconds.
            compute_rates: Estimate respiratory and heart rates.
            compute_breath_intervals: Detect and store EIT breath intervals.
            compute_continuous_tiv: Compute a tidal impedance variation per breath.
            compute_eeli: Compute end-expiratory lung impedance per breath.
            compute_pixel_tiv: Compute a pixel-TIV map per breath.
            include_filtered_data: Include the filtered EIT object in the return mapping.
            include_global_impedance: Obtain raw global impedance and, after filtering,
                filtered global impedance. Breath calculations obtain global impedance
                as needed even when this option is False.

        Returns:
            dict[str, Any]: The sequence, raw/filtered EIT and global impedance,
                filter mode and captures, rate detector and captures, respiratory
                and heart rates in Hz, breath intervals, TIV, EELI and pixel TIV.
                Disabled optional results are None; capture mappings are empty when
                their calculation is skipped. Impedance results retain input units.

        Raises:
            OptionalDependencyError: If eitprocessing is unavailable.
            TypeError: If sequence does not provide the expected EIT collections.
            KeyError: If sequence.eit_data has no raw EIT entry.
            ValueError: If the filter mode is unsupported, or TIV/EELI/pixel TIV
                is requested while breath-interval calculation is disabled.
        """

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
            filtered_pixels = filter_pixels_preserving_gaps(
                raw_eit.pixel_impedance,
                operation=f"preprocess(filter_mode={normalized_filter_mode!r})",
                apply=lambda pixels: butterworth_filter(
                    pixels,
                    filter_type=butterworth_filter_type,
                    cutoff_frequency=cutoff_frequency,
                    sample_frequency=raw_eit.sample_frequency,
                    order=filter_order,
                    axis=0,
                    captures=filter_captures,
                ),
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
                    processing_state="intermediate",
                    method=f"eitprocessing.{preprocessed.get('filter_mode')}_filter",
                    name=getattr(filtered_gi, "name", None)
                    or getattr(filtered_gi, "label", None),
                )
            )
        return signals

    def to_parameters(self, preprocessed: dict[str, Any]) -> list[ParameterResult]:
        """Convert available respiratory and heart rates to ``ParameterResult``.

        Args:
            preprocessed: Preprocessing output with optional
                ``respiratory_rate_hz`` and ``heart_rate_hz`` entries, in Hz.

        Returns:
            list[ParameterResult]: Available numeric rates in respiratory-rate,
                heart-rate order, tagged with unit ``Hz``. Missing entries
                are omitted.

        Raises:
            ValueError: If a supplied rate cannot be converted to a number.
            TypeError: If a supplied rate has an unsupported type.
        """

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
        return parameters

    def to_interval_data(
        self,
        preprocessed: dict[str, Any],
        *,
        stored_breaths: Iterable[Any] | None = None,
    ) -> list[IntervalData]:
        """Convert per-breath TIV, EELI and pixel TIV into ``IntervalData``.

        Args:
            preprocessed: Preprocessing output with optional ``continuous_tiv``,
                ``eeli`` and ``pixel_tiv`` results and their ``breath_intervals``.
                Each result must have one value per detected breath, in the
                same order. Units are taken from each result.
            stored_breaths: Existing breaths to reuse when modality and exact
                start and end times match. ``None`` uses newly converted breaths.

        Returns:
            list[IntervalData]: Available results in TIV, EELI, pixel-TIV
                order, sharing their breath objects. Pixel TIV values are
                ``PixelMap`` objects with row-column grids. An empty list is
                returned when all three results are absent.

        Raises:
            ValueError: If results lack ``breath_intervals``, value and breath
                counts differ, or pixel grids have other than two dimensions.
        """

        per_breath = [
            (preprocessed[key], method, as_pixel_maps)
            for key, method, as_pixel_maps in (
                ("continuous_tiv", "eitprocessing.TIV", False),
                ("eeli", "eitprocessing.EELI", False),
                ("pixel_tiv", "eitprocessing.TIV", True),
            )
            if preprocessed.get(key) is not None
        ]
        if not per_breath:
            return []
        breath_intervals = preprocessed.get("breath_intervals")
        if breath_intervals is None:
            raise ValueError(
                "TIV/EELI values were given without the breaths they were "
                "computed over ('breath_intervals')."
            )
        breaths = reuse_matching_breaths(
            breath_intervals_to_breath_events(breath_intervals), stored_breaths
        )

        results: list[IntervalData] = []
        for sparse, method, as_pixel_maps in per_breath:
            results.append(
                sparse_data_to_interval_data(
                    sparse,
                    breaths,
                    modality="eit",
                    method=method,
                    as_pixel_maps=as_pixel_maps,
                )
            )
        return results

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
