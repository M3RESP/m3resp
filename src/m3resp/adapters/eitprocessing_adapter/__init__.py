"""Load and process EIT recordings through eitprocessing.

Conversion helpers produce m3resp signals, breath events, rate parameters and
values per breath from eitprocessing results.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import (
    IntervalData,
    ParameterResult,
    PixelBreaths,
    PixelMask,
    QualityFlag,
    Signal,
)
from m3resp.data.events import (
    BreathEvent,
    coerce_breath_events,
    reuse_matching_breaths,
)
from m3resp.eit.io import EITFileData, join_eit_files, read_eit_file
from m3resp.eit.processing import (
    BreathDetectionSettings,
    RateDetectionSettings,
    end_expiratory_lung_impedance,
    global_impedance,
    mdn_filter,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
    tidal_impedance_variation,
)
from m3resp.eit.processing import detect_rates as detect_rates_of
from m3resp.eit.processing import find_breaths as find_breaths_of
from m3resp.eit.processing import find_pixel_breaths as find_pixel_breaths_of
from m3resp.eit.processing.mdn_filter import (
    NOISE_FREQUENCY_LIMIT,
    NOTCH_DISTANCE,
    ORDER,
)
from m3resp.eit.roi import (
    amplitude_lungspace,
    filter_mask_by_size,
    tiv_lungspace,
    watershed_lungspace,
)

from ._file_data import (
    file_data_to_sequence,
    load_eit_file_as_sequence,
    sequence_to_file_data,
)
from ._filtering import butterworth_filter_images
from ._joining import eit_file_paths, join_eit_sequences
from ._shared import (
    MAX_EIT_BUTTERWORTH_ORDER,
    _lazy_import,
    _require_eit_sequence,
    add_to_collection,
    as_breath_detection,
    breath_intervals_to_breath_events,
    breaths_to_eitprocessing,
    breaths_to_interval_data,
    check_eit_butterworth_order,
    filter_pixels_preserving_gaps,
    given_breath_events,
    pixel_sign_breaths,
    sparse_data_to_interval_data,
)
from ._signals import (
    EITPROCESSING_FIELDS,
    as_continuous_data,
    as_eit_data,
    as_eitprocessing,
    as_signal,
    continuous_data_fields,
    continuous_data_to_signal,
    eit_data_fields,
    eit_data_to_signal,
    eitprocessing_fields,
    signal_to_continuous_data,
    signal_to_eit_data,
)

__all__ = [
    "EITPROCESSING_FIELDS",
    "MAX_EIT_BUTTERWORTH_ORDER",
    "EITProcessingAdapter",
    "add_to_collection",
    "as_breath_detection",
    "as_continuous_data",
    "as_eit_data",
    "as_eitprocessing",
    "as_signal",
    "breath_intervals_to_breath_events",
    "breaths_to_eitprocessing",
    "breaths_to_interval_data",
    "butterworth_filter_images",
    "check_eit_butterworth_order",
    "continuous_data_fields",
    "continuous_data_to_signal",
    "eit_data_fields",
    "eit_data_to_signal",
    "eitprocessing_fields",
    "file_data_to_sequence",
    "filter_pixels_preserving_gaps",
    "given_breath_events",
    "load_eit_file_as_sequence",
    "pixel_sign_breaths",
    "sequence_to_file_data",
    "signal_to_continuous_data",
    "signal_to_eit_data",
    "sparse_data_to_interval_data",
]


def _pixel_or_global_breaths(
    breaths: Any, *, function: str
) -> list[BreathEvent] | PixelBreaths:
    """Return the breaths of a pixel TIV or pixel EELI calculation.

    Args:
        breaths (Any): A ``PixelBreaths`` (pixel timing), or global breaths
            in a form of ``given_breath_events``.
        function (str): Name of the calling function, for the message.

    Returns:
        list[BreathEvent] | PixelBreaths: A ``PixelBreaths`` as it is; other
            breaths as a list of ``BreathEvent``s.
    """

    if isinstance(breaths, PixelBreaths):
        return breaths
    return given_breath_events(breaths, function=function)


def _lungspace_inputs(
    eit_data: Any, timing_data: Any, breaths: Any, *, function: str
) -> tuple[Signal, Signal, list[BreathEvent]]:
    """Return the images, the waveform and the global breaths of a lung space.

    Args:
        eit_data (Any): EIT images as a ``Signal`` or an eitprocessing
            ``EITData``.
        timing_data (Any): The impedance waveform, or None for the sum of all
            pixels (eitprocessing: ``EITData.get_summed_impedance``).
        breaths (Any): Given global breaths in a form of
            ``given_breath_events``, or None to find them on the waveform
            with the default ``BreathDetectionSettings``, as eitprocessing
            does.
        function (str): Name of the calling function, for the message.

    Returns:
        tuple[Signal, Signal, list[BreathEvent]]: The images, the waveform
            and the breaths.
    """

    pixels = as_signal(eit_data)
    waveform = (
        global_impedance(pixels) if timing_data is None else as_signal(timing_data)
    )
    if breaths is None:
        return pixels, waveform, find_breaths_of(waveform, BreathDetectionSettings())
    return pixels, waveform, given_breath_events(breaths, function=function)


def _as_m3resp_pixel_mask(mask: Any) -> PixelMask:
    """Return a mask as an m3resp ``PixelMask``.

    Args:
        mask (Any): An m3resp ``PixelMask`` (returned as it is), an
            eitprocessing ``PixelMask`` (its grid is copied, weights and
            zeros included), or a 2D numeric array, list or tuple with NaN
            for excluded pixels (a 0 becomes NaN, as in eitprocessing).

    Returns:
        PixelMask: The m3resp mask.

    Raises:
        UnsupportedWorkflowError: If the input has another type or is not a
            2D numeric grid.
    """

    if isinstance(mask, PixelMask):
        return mask
    if hasattr(mask, "mask"):
        return PixelMask(
            name=str(getattr(mask, "label", None) or "mask"),
            values=np.asarray(mask.mask, dtype=float),
            keep_zeros=True,
            suppress_value_range_error=True,
            suppress_all_nan_warning=True,
        )
    if not isinstance(mask, (np.ndarray, list, tuple)):
        raise UnsupportedWorkflowError(
            "An ROI mask must be an m3resp PixelMask, an eitprocessing "
            f"PixelMask, or a 2D array of pixels; got {type(mask).__name__}."
        )
    try:
        array = np.asarray(mask, dtype=float)
    except (TypeError, ValueError) as error:
        raise UnsupportedWorkflowError(
            "An ROI mask must be a 2D (row, column) array of numbers."
        ) from error
    if array.ndim != 2:
        raise UnsupportedWorkflowError(
            "An ROI mask must be a 2D (row, column) array of pixels; got "
            f"shape {array.shape}."
        )
    return PixelMask(name="mask", values=array)


class EITProcessingAdapter:
    """Thin wrapper around `eitprocessing`.

    It imports `eitprocessing` only when used, so `m3resp` can be installed
    without optional EIT support.
    """

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    @property
    def has_custom_loader(self) -> bool:
        """True when this adapter was given a loader of its own (``loader=``)."""

        return self._loader is not None

    def load(self, path: str, vendor: str | None = None, **kwargs: Any) -> Any:
        """Load EIT data through the injected loader, or else `eitprocessing`."""

        if self._loader is not None:
            return self._loader(path, vendor=vendor, **kwargs)
        return self.load_with_eitprocessing(path, vendor=vendor, **kwargs)

    def load_with_custom_loader(self, path: str, **kwargs: Any) -> Any:
        """Load EIT data through the loader given as ``loader=``.

        Args:
            path: File (or folder) the loader reads.
            **kwargs: Passed on to the loader.

        Returns:
            Whatever the loader returns, normally an eitprocessing Sequence.

        Raises:
            UnsupportedWorkflowError: If this adapter has no loader of its own.
        """

        if self._loader is None:
            raise UnsupportedWorkflowError(
                "No custom EIT loader is set. Create the session with "
                "M3Session(eit_adapter=EITProcessingAdapter(loader=my_loader)), "
                "or pass that adapter as run_workflow(..., eit_adapter=...). "
                "Use 'eit.load' with a vendor for Draeger, Sentec or Timpel files."
            )
        return self._loader(path, vendor=None, **kwargs)

    def load_with_eitprocessing(
        self, path: str, vendor: str | None = None, **kwargs: Any
    ) -> Any:
        """Load a Draeger, Sentec or Timpel file into an eitprocessing Sequence.

        The same as `load_with_native_data`, without the m3resp data.

        Args:
            path: The EIT file to read, or a list of files recorded one after
                the other.
            vendor: ``"draeger"``, ``"sentec"`` or ``"timpel"``.
            **kwargs: See `load_with_native_data`.

        Returns:
            The loaded eitprocessing Sequence.

        Raises:
            ValueError: See `load_with_native_data`.
            OptionalDependencyError: If eitprocessing is not installed.
        """

        sequence, _ = self.load_with_native_data(path, vendor=vendor, **kwargs)
        return sequence

    def load_with_native_data(
        self, path: str, vendor: str | None = None, **kwargs: Any
    ) -> tuple[Any, EITFileData]:
        """Load a Draeger, Sentec or Timpel file as m3resp data and as a Sequence.

        Each file is read by the m3resp reader of its vendor
        (`m3resp.eit.io.read_eit_file`). Several files are read as one
        recording: the m3resp data of the files is joined in the given order
        (`m3resp.eit.io.join_eit_files`). The eitprocessing Sequence of each
        file is built with `file_data_to_sequence`, and the Sequences are
        joined with `join_eit_sequences`, so the Sequence is the one
        eitprocessing's ``load_eit_data`` gives.

        Args:
            path: The EIT file to read, or a list of files recorded one after
                the other.
            vendor: ``"draeger"``, ``"sentec"`` or ``"timpel"``.
            **kwargs: ``sample_frequency``, ``first_frame`` and
                ``max_frames`` for the reader; ``label``, ``name`` and
                ``description`` for the Sequence.

        Returns:
            tuple[Any, EITFileData]: The eitprocessing Sequence, and the
            m3resp data of all files joined (its ``path`` is the first file).

        Raises:
            ValueError: If vendor is None; if ``first_frame`` or
                ``max_frames`` is set with more than one file (they would
                apply to each file); or if the files cannot be joined.
            TypeError: If a keyword argument is unknown.
            OptionalDependencyError: If eitprocessing is not installed.
        """

        if vendor is None:
            raise ValueError(
                "EIT loading requires an explicit vendor (e.g. 'draeger', "
                "'sentec', or 'timpel') when no loader is injected."
            )

        _lazy_import(
            "eitprocessing.datahandling.sequence.Sequence",
            error=lambda: OptionalDependencyError(
                "EIT support requires the optional dependency `eitprocessing`. "
                'Install with `pip install "m3resp[eit]"` or inject a loader.'
            ),
        )

        sequence_options = {
            key: kwargs.pop(key)
            for key in ("label", "name", "description")
            if key in kwargs
        }
        unknown = set(kwargs) - {"sample_frequency", "first_frame", "max_frames"}
        if unknown:
            raise TypeError(
                f"Unknown EIT loading option(s) {sorted(unknown)}. Known: "
                "sample_frequency, first_frame, max_frames, label, name, "
                "description."
            )

        paths = eit_file_paths(path)
        if len(paths) > 1 and (
            kwargs.get("first_frame") or kwargs.get("max_frames") is not None
        ):
            raise ValueError(
                "first_frame and max_frames apply to each file, so they cannot "
                "be used when loading several EIT files at once. Load the "
                "files, then cut the recording with eit.slice_recording "
                "(session.slice_eit)."
            )
        files = [read_eit_file(single_path, vendor, **kwargs) for single_path in paths]
        native = join_eit_files(files, paths)
        sequences = [
            file_data_to_sequence(file_data, **sequence_options) for file_data in files
        ]
        if len(sequences) == 1:
            return sequences[0], native
        return join_eit_sequences(sequences, paths, check_order=False), native

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
        masked = upstream_mask.apply(as_eit_data(eit_data))
        return masked.get_summed_impedance(
            return_label=label,
            name=label.replace("_", " ").capitalize(),
            description="Impedance summed over the pixels of a mask.",
        )

    def slice_sequence(self, sequence: Any, start_index: int, end_index: int) -> Any:
        """Return a copy of `sequence` keeping frames `start_index` up to (not
        including) `end_index`.

        m3resp cuts the native images, waveforms, device marks and breaths,
        then rebuilds a Sequence with the original collection keys and field
        spellings. Frame times keep their original values.
        """

        from ._slicing import slice_sequence

        return slice_sequence(sequence, start_index, end_index)

    # -- Phase 1: reusable adapter operations ----------------------------------
    #
    # Small public methods, each running one calculation: an `eitprocessing`
    # operation, or its m3resp port (`detect_rates`, `find_breaths`).
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
        """Find the breathing rate and the heart rate of EIT images.

        Args:
            signal: EIT images as a ``Signal`` with values of shape (frames,
                rows, columns), or an eitprocessing ``EITData``.
            subject_type: ``"adult"`` or ``"neonate"``; sets the rate ranges.
            welch_window_seconds: Length of one Welch window in seconds.
                None gives 30 s.
            capture: If True, also return the spectra for the rate figure and
                give no warnings about short data or a rate at the edge of
                its range.

        Returns:
            dict[str, Any]: ``respiratory_rate_hz`` and ``heart_rate_hz`` in
                Hz, ``rate_detector`` (the ``RateDetectionSettings``) and
                ``rate_captures`` (the spectra, empty when ``capture`` is
                False).

        Raises:
            ValueError: If a setting is invalid, or the images cannot give a
                rate (see ``m3resp.eit.processing.detect_rates``).
        """

        settings = (
            RateDetectionSettings(subject_type)
            if welch_window_seconds is None
            else RateDetectionSettings(subject_type, welch_window=welch_window_seconds)
        )
        captures: dict[str, Any] = {}
        respiratory_rate_hz, heart_rate_hz = detect_rates_of(
            as_signal(signal),
            settings,
            captures=captures if capture else None,
            suppress_length_warnings=capture,
            suppress_edge_case_warning=capture,
        )
        return {
            "respiratory_rate_hz": respiratory_rate_hz,
            "heart_rate_hz": heart_rate_hz,
            "rate_detector": settings,
            "rate_captures": captures,
        }

    def apply_mdn(
        self,
        signal: Any,
        *,
        respiratory_rate_hz: float,
        heart_rate_hz: float,
        noise_frequency_limit_hz: float = NOISE_FREQUENCY_LIMIT,
        notch_distance_hz: float = NOTCH_DISTANCE,
        order: int = ORDER,
        label: str = "filtered",
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Remove the heart signal from EIT images with the MDN filter.

        Args:
            signal (Any): EIT images as a ``Signal`` (time along the first
                axis), or an eitprocessing ``EITData``.
            respiratory_rate_hz (float): Breathing rate in Hz.
            heart_rate_hz (float): Heart rate in Hz.
            noise_frequency_limit_hz (float): Low-pass edge in Hz. Defaults
                to 220 per minute.
            notch_distance_hz (float): Half the width of each removed band,
                in Hz. Defaults to 10 per minute.
            order (int): Butterworth order of each filter. Defaults to 10.
            label (str): Label of the filtered images in an eitprocessing
                ``Sequence``.
            name (str | None): Name of the filtered images. None keeps the
                name of ``signal``.
            description (str | None): Description of the filtered images in
                an eitprocessing ``Sequence``. None keeps the description of
                ``signal``.

        Returns:
            dict[str, Any]: ``filtered_eit``, the filtered images as a
                ``Signal`` (``signal_to_eit_data`` gives the eitprocessing
                ``EITData`` with ``label``, ``name`` and ``description``), and
                ``filter_captures``, the in-between results of
                ``m3resp.eit.processing.mdn_filter``.

        Raises:
            ValueError: If a rate is not a finite number above 0, or if
                ``mdn_filter`` refuses the rates or the settings.
        """

        for rate_name, rate in (
            ("respiratory_rate_hz", respiratory_rate_hz),
            ("heart_rate_hz", heart_rate_hz),
        ):
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(
                    f"{rate_name} must be a finite, positive value in Hz, got {rate!r}."
                )

        images = as_signal(signal)
        captures: dict[str, Any] = {}
        filtered_eit = mdn_filter(
            images,
            respiratory_rate=respiratory_rate_hz,
            heart_rate=heart_rate_hz,
            noise_frequency_limit=noise_frequency_limit_hz,
            notch_distance=notch_distance_hz,
            order=order,
            name=name,
            captures=captures,
        )
        fields = {
            **eitprocessing_fields(images),
            "type": "EITData",
            "label": label,
            "name": filtered_eit.name,
        }
        if description is not None:
            fields["description"] = description
        filtered_eit.metadata[EITPROCESSING_FIELDS] = fields
        return {"filtered_eit": filtered_eit, "filter_captures": captures}

    def find_breaths(
        self, timing_data: Any, *, minimum_duration_seconds: float = 2 / 3
    ) -> IntervalData:
        """Find the breaths of a global or regional impedance waveform.

        Args:
            timing_data: The waveform as a ``Signal``, or an eitprocessing
                ``ContinuousData``, with time in seconds.
            minimum_duration_seconds: Shortest expected breath, in seconds.

        Returns:
            IntervalData: The breaths as ``BreathEvent``s (start, peak in
                ``extremum_time``, end) on the time axis of the waveform.
        """

        settings = BreathDetectionSettings(minimum_duration=minimum_duration_seconds)
        return breaths_to_interval_data(
            find_breaths_of(as_signal(timing_data), settings)
        )

    def find_pixel_breaths(
        self,
        eit_data: Any,
        timing_data: Any,
        *,
        breaths: Any = None,
        phase_correction_mode: Literal["negative amplitude", "phase shift", "none"]
        | None = "negative amplitude",
        minimum_duration_seconds: float = 2 / 3,
        result_label: str = "pixel_breaths",
    ) -> PixelBreaths:
        """Find the breaths of each pixel of EIT images.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            timing_data (Any): The global or regional impedance waveform as
                a ``Signal``, or an eitprocessing ``ContinuousData``, on the
                time axis of ``eit_data``.
            breaths (Any): The global breaths on the time axis of the
                waveform: a list of ``BreathEvent``s, an m3resp
                ``IntervalData`` of them, or an eitprocessing
                ``IntervalData`` of ``Breath``s. None finds them on
                ``timing_data``.
            phase_correction_mode (str | None): ``"negative amplitude"``,
                ``"phase shift"``, ``"none"`` or None.
            minimum_duration_seconds (float): Shortest expected breath, in
                seconds, for the breaths found on ``timing_data``.
            result_label (str): Name of the result.

        Returns:
            PixelBreaths: The result of
                ``m3resp.eit.processing.find_pixel_breaths``. For breaths
                found on ``timing_data``, the sign of each pixel comes from
                the breaths found with the default settings, as in
                eitprocessing. Given breaths give the sign themselves.

        Raises:
            TypeError: If a breath of a list or of an m3resp ``IntervalData``
                is not a ``BreathEvent``.
            ValueError: If ``find_pixel_breaths`` refuses the data, the
                breaths or the phase correction mode.
        """

        waveform = as_signal(timing_data)
        sign_breaths = None
        if breaths is None:
            settings = BreathDetectionSettings(
                minimum_duration=minimum_duration_seconds
            )
            global_breaths = find_breaths_of(waveform, settings)
            sign_breaths = pixel_sign_breaths(waveform, settings)
        else:
            global_breaths = given_breath_events(breaths, function="find_pixel_breaths")
        return find_pixel_breaths_of(
            as_signal(eit_data),
            waveform,
            global_breaths,
            phase_correction_mode=phase_correction_mode,
            name=result_label,
            sign_breaths=sign_breaths,
        )

    def compute_continuous_tiv(
        self,
        timing_data: Any,
        *,
        breaths: Any,
        tiv_method: str = "inspiratory",
        result_label: str = "continuous_tivs",
    ) -> IntervalData:
        """Compute the tidal impedance variation (TIV) of each breath.

        Args:
            timing_data: The impedance waveform as a ``Signal``, or an
                eitprocessing ``ContinuousData``, with time in seconds.
            breaths: The breaths on the time axis of the waveform: a list of
                ``BreathEvent``s, an m3resp ``IntervalData`` of them, or an
                eitprocessing ``IntervalData`` of ``Breath``s.
            tiv_method: ``"inspiratory"``, ``"expiratory"`` or ``"mean"``.
            result_label: Name of the result.

        Returns:
            IntervalData: One TIV value per breath, from
                ``m3resp.eit.processing.tidal_impedance_variation``.

        Raises:
            TypeError: If a breath of a list or of an m3resp ``IntervalData``
                is not a ``BreathEvent``.
        """

        return tidal_impedance_variation(
            as_signal(timing_data),
            given_breath_events(breaths, function="compute_continuous_tiv"),
            method=tiv_method,
            name=result_label,
        )

    def compute_eeli(
        self,
        timing_data: Any,
        *,
        breaths: Any,
        result_label: str = "continuous_eelis",
    ) -> IntervalData:
        """Compute the end-expiratory lung impedance (EELI) of each breath.

        Args:
            timing_data: The impedance waveform as a ``Signal``, or an
                eitprocessing ``ContinuousData``, with time in seconds.
            breaths: The breaths on the time axis of the waveform: a list of
                ``BreathEvent``s, an m3resp ``IntervalData`` of them, or an
                eitprocessing ``IntervalData`` of ``Breath``s.
            result_label: Name of the result.

        Returns:
            IntervalData: One EELI value per breath, from
                ``m3resp.eit.processing.end_expiratory_lung_impedance``.

        Raises:
            TypeError: If a breath of a list or of an m3resp ``IntervalData``
                is not a ``BreathEvent``.
        """

        return end_expiratory_lung_impedance(
            as_signal(timing_data),
            given_breath_events(breaths, function="compute_eeli"),
            name=result_label,
        )

    def compute_pixel_tiv(
        self,
        eit_data: Any,
        *,
        breaths: Any,
        tiv_method: str = "inspiratory",
        result_label: str = "pixel_tivs",
    ) -> IntervalData:
        """Compute the tidal impedance variation (TIV) of each pixel and breath.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            breaths (Any): The breaths on the time axis of the images.
                Global timing: a list of ``BreathEvent``s, an m3resp
                ``IntervalData`` of them, or an eitprocessing
                ``IntervalData`` of ``Breath``s. Pixel timing: a
                ``PixelBreaths`` (see ``find_pixel_breaths``).
            tiv_method (str): ``"inspiratory"``, ``"expiratory"`` or
                ``"mean"``.
            result_label (str): Name of the result.

        Returns:
            IntervalData: One ``PixelMap`` of TIV per breath, from
                ``m3resp.eit.processing.pixel_tidal_impedance_variation``.

        Raises:
            TypeError: If a breath of a list or of an m3resp ``IntervalData``
                is not a ``BreathEvent``.
        """

        return pixel_tidal_impedance_variation(
            as_signal(eit_data),
            _pixel_or_global_breaths(breaths, function="compute_pixel_tiv"),
            method=tiv_method,
            name=result_label,
        )

    def compute_pixel_eeli(
        self,
        eit_data: Any,
        *,
        breaths: Any,
        result_label: str = "pixel_eelis",
    ) -> IntervalData:
        """Compute the end-expiratory lung impedance (EELI) of each pixel and breath.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            breaths (Any): The breaths on the time axis of the images, in
                one of the forms of ``compute_pixel_tiv``.
            result_label (str): Name of the result.

        Returns:
            IntervalData: One ``PixelMap`` of EELI per breath, from
                ``m3resp.eit.processing.pixel_end_expiratory_lung_impedance``.

        Raises:
            TypeError: If a breath of a list or of an m3resp ``IntervalData``
                is not a ``BreathEvent``.
        """

        return pixel_end_expiratory_lung_impedance(
            as_signal(eit_data),
            _pixel_or_global_breaths(breaths, function="compute_pixel_eeli"),
            name=result_label,
        )

    def compute_tiv_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        breaths: Any = None,
        threshold: float = 0.15,
    ) -> dict[str, Any]:
        """Find the functional lung space from the mean TIV of each pixel.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            timing_data (Any | None): The impedance waveform of the breaths
                as a ``Signal``, or an eitprocessing ``ContinuousData``.
                None uses the sum of all pixels.
            breaths (Any): The global breaths on the time axis of the
                images: a list of ``BreathEvent``s, an m3resp
                ``IntervalData`` of them, or an eitprocessing
                ``IntervalData`` of ``Breath``s. None finds them on
                ``timing_data`` with the default settings, as in
                eitprocessing.
            threshold (float): Fraction of the largest mean TIV, between 0
                and 1.

        Returns:
            dict[str, Any]: ``"mask"``, the m3resp ``PixelMask`` of
                ``m3resp.eit.roi.tiv_lungspace``, and ``"captures"``, with
                ``"TIV per breath"`` and ``"mean TIV"``.

        Raises:
            TypeError: If ``threshold`` is not a float, or a given breath is
                not a ``BreathEvent``.
            ValueError: If ``threshold`` is not between 0 and 1, there are no
                breaths, or every TIV value is NaN.
        """

        pixels, _, found = _lungspace_inputs(
            eit_data, timing_data, breaths, function="compute_tiv_lungspace"
        )
        captures: dict[str, Any] = {}
        mask = tiv_lungspace(pixels, found, threshold=threshold, captures=captures)
        return {"mask": mask, "captures": captures}

    def compute_amplitude_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        breaths: Any = None,
        threshold: float = 0.15,
    ) -> dict[str, Any]:
        """Find a lung space from the mean amplitude of each pixel.

        This lung space can include reconstruction artefacts, so it is not
        advised on its own; ``compute_watershed_lungspace`` uses the
        amplitude together with the TIV.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            timing_data (Any | None): The impedance waveform of the breaths,
                for the phase lag of each pixel, as a ``Signal`` or an
                eitprocessing ``ContinuousData``. None uses the sum of all
                pixels.
            breaths (Any): The global EIT breaths, in one of the forms of
                ``compute_tiv_lungspace``. None finds them on
                ``timing_data`` with the default settings, as in
                eitprocessing.
            threshold (float): Fraction of the largest mean amplitude,
                between 0 and 1.

        Returns:
            dict[str, Any]: ``"mask"``, the m3resp ``PixelMask`` of
                ``m3resp.eit.roi.amplitude_lungspace``, and ``"captures"``,
                with ``"amplitude per breath"`` and ``"mean amplitude"``.

        Raises:
            TypeError: If ``threshold`` is not a float, or a given breath is
                not a ``BreathEvent``.
            ValueError: If ``threshold`` is not between 0 and 1, there are no
                breaths, or every amplitude value is NaN.
        """

        pixels, waveform, found = _lungspace_inputs(
            eit_data, timing_data, breaths, function="compute_amplitude_lungspace"
        )
        captures: dict[str, Any] = {}
        mask = amplitude_lungspace(
            pixels, waveform, found, threshold=threshold, captures=captures
        )
        return {"mask": mask, "captures": captures}

    def compute_watershed_lungspace(
        self,
        eit_data: Any,
        *,
        timing_data: Any | None = None,
        breaths: Any = None,
        threshold_fraction: float = 0.15,
    ) -> dict[str, Any]:
        """Find the functional lung space with the watershed method.

        Args:
            eit_data (Any): EIT images as a ``Signal``, or an eitprocessing
                ``EITData``, with time, row and column axes.
            timing_data (Any | None): The impedance waveform of the breaths,
                for the phase lag of each pixel, as a ``Signal`` or an
                eitprocessing ``ContinuousData``. None uses the sum of all
                pixels.
            breaths (Any): The global EIT breaths, in one of the forms of
                ``compute_tiv_lungspace``. None finds them on
                ``timing_data`` with the default settings, as in
                eitprocessing.
            threshold_fraction (float): Fraction of the largest mean TIV for
                the TIV mask, between 0 and 1.

        Returns:
            dict[str, Any]: ``"mask"``, the m3resp ``PixelMask`` of
                ``m3resp.eit.roi.watershed_lungspace``, and ``"captures"``
                with its intermediate results.

        Raises:
            TypeError: If ``threshold_fraction`` is not a float, or a given
                breath is not a ``BreathEvent``.
            ValueError: If ``threshold_fraction`` is not between 0 and 1, no
                breath has both TIV and amplitude values, or no local maximum
                of the mean amplitude lies inside the TIV mask.
        """

        pixels, waveform, found = _lungspace_inputs(
            eit_data, timing_data, breaths, function="compute_watershed_lungspace"
        )
        captures: dict[str, Any] = {}
        mask = watershed_lungspace(
            pixels,
            waveform,
            found,
            threshold_fraction=threshold_fraction,
            captures=captures,
        )
        return {"mask": mask, "captures": captures}

    def filter_roi_by_size(
        self,
        mask: Any,
        *,
        min_region_size: int = 10,
        connectivity: Literal[1, 2] | np.ndarray = 1,
    ) -> PixelMask:
        """Keep the groups of connected mask pixels with a minimum number of pixels.

        Args:
            mask (Any): An m3resp or eitprocessing ``PixelMask``, or a 2D
                numeric grid with NaN for excluded pixels.
            min_region_size (int): Smallest number of pixels of a kept
                group.
            connectivity (int | numpy.ndarray): ``1`` joins pixels that
                share an edge; ``2`` also joins pixels that share a corner.
                A custom structure array is accepted.

        Returns:
            PixelMask: The m3resp mask of
                ``m3resp.eit.roi.filter_mask_by_size``, with 1 for the kept
                pixels and NaN elsewhere.

        Raises:
            UnsupportedWorkflowError: If the mask cannot be converted to a
                2D numeric grid.
            ValueError: If ``connectivity`` is not 1, 2 or an array.
            RuntimeError: If no group has ``min_region_size`` pixels or more.
        """

        return filter_mask_by_size(
            _as_m3resp_pixel_mask(mask),
            min_region_size=min_region_size,
            connectivity=connectivity,
        )

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
        compute_pixel_eeli: bool = False,
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
            filter_order: Butterworth filter order for lowpass or bandpass, 1 to 10.
            breath_min_duration_seconds: Minimum detected breath duration in seconds.
            compute_rates: Estimate respiratory and heart rates.
            compute_breath_intervals: Detect and store EIT breath intervals.
            compute_continuous_tiv: Compute a tidal impedance variation per breath.
            compute_eeli: Compute end-expiratory lung impedance per breath.
            compute_pixel_tiv: Compute a pixel-TIV map per breath.
            compute_pixel_eeli: Compute a pixel-EELI map per breath.
            include_filtered_data: Include the filtered EIT object in the return mapping.
            include_global_impedance: Obtain raw global impedance and, after filtering,
                filtered global impedance. Breath calculations obtain global impedance
                as needed even when this option is False.

        Returns:
            dict[str, Any]: The sequence, raw/filtered EIT and global impedance,
                filter mode and captures, rate settings (``rate_detector``) and
                captures, respiratory and heart rates in Hz, the breaths
                (``breath_intervals``, an m3resp ``IntervalData`` of
                ``BreathEvent``s; the ``Sequence`` also gets them as
                ``"eit_breaths"``), TIV and EELI (m3resp ``IntervalData``,
                one value per breath of ``breath_intervals``), and pixel TIV
                and pixel EELI (m3resp ``IntervalData``, one ``PixelMap``
                per breath of ``breath_intervals``, with the timing of the
                global breaths). Results that are switched off are None;
                the captures are empty when their calculation is skipped.
                Impedance results keep the unit of the input.

        Raises:
            OptionalDependencyError: If eitprocessing is unavailable.
            TypeError: If sequence does not provide the expected EIT collections.
            KeyError: If sequence.eit_data has no raw EIT entry.
            ValueError: If the filter mode is unsupported, if the Butterworth order
                is outside 1 to 10, or TIV, EELI, pixel TIV or pixel EELI is
                requested while breath-interval calculation is disabled.
        """

        # The Sequence and the filtered images in it need eitprocessing.
        _lazy_import(
            "eitprocessing.datahandling.sequence.Sequence",
            error=lambda: OptionalDependencyError(
                "EIT preprocessing requires the optional dependency "
                '`eitprocessing`. Install with `pip install "m3resp[eit]"`.'
            ),
        )

        _require_eit_sequence(sequence)

        if not compute_breath_intervals and (
            compute_continuous_tiv
            or compute_eeli
            or compute_pixel_tiv
            or compute_pixel_eeli
        ):
            raise ValueError(
                "EIT TIV, EELI, pixel TIV and pixel EELI require breath intervals. "
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
            # The filtered images go into the Sequence, so they go on as an
            # eitprocessing ``EITData``.
            filtered_eit = signal_to_eit_data(mdn_result["filtered_eit"])
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
            filtered_images, filter_captures = butterworth_filter_images(
                as_signal(raw_eit),
                mode=butterworth_filter_type,
                cutoff_frequency=cutoff_frequency,
                order=filter_order,
                label=f"{normalized_filter_mode}_filtered",
                operation=f"preprocess(filter_mode={normalized_filter_mode!r})",
            )
            filtered_eit = signal_to_eit_data(filtered_images)

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
            breath_detector = BreathDetectionSettings(
                minimum_duration=breath_min_duration_seconds
            )
            breaths = find_breaths_of(
                as_signal(filtered_global_impedance), breath_detector
            )
            breath_intervals = breaths_to_interval_data(breaths)
            add_to_collection(
                sequence.interval_data,
                breaths_to_eitprocessing(
                    breaths,
                    breath_detector,
                    label="eit_breaths",
                    derived_from=[filtered_global_impedance],
                ),
            )

        # `compute_breath_intervals` is guaranteed True here (validated above
        # whenever any of these four outputs is requested), so
        # `breath_intervals`, `breath_detector` and `filtered_global_impedance` are set;
        # the asserts only narrow the type for mypy.
        if compute_continuous_tiv:
            assert breath_intervals is not None
            assert filtered_global_impedance is not None
            continuous_tiv = self.compute_continuous_tiv(
                filtered_global_impedance, breaths=breath_intervals
            )

        if compute_eeli:
            assert breath_intervals is not None
            assert filtered_global_impedance is not None
            eeli = self.compute_eeli(
                filtered_global_impedance, breaths=breath_intervals
            )

        pixel_tiv = None
        if compute_pixel_tiv:
            assert breath_intervals is not None
            pixel_tiv = self.compute_pixel_tiv(
                filtered_eit, breaths=breath_intervals, result_label="pixel_tivs"
            )

        pixel_eeli = None
        if compute_pixel_eeli:
            assert breath_intervals is not None
            pixel_eeli = self.compute_pixel_eeli(
                filtered_eit, breaths=breath_intervals, result_label="pixel_eelis"
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
            "pixel_eeli": pixel_eeli,
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
            return breath_intervals_to_breath_events(data["breath_intervals"])

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
                    method="m3resp.eit.processing.detect_rates",
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
                    method="m3resp.eit.processing.detect_rates",
                )
            )
        return parameters

    def to_interval_data(
        self,
        preprocessed: dict[str, Any],
        *,
        stored_breaths: Iterable[Any] | None = None,
    ) -> list[IntervalData]:
        """Convert per-breath TIV, EELI, pixel TIV and pixel EELI into ``IntervalData``.

        Args:
            preprocessed: Preprocessing output with optional ``continuous_tiv``,
                ``eeli``, ``pixel_tiv`` and ``pixel_eeli`` results and their
                ``breath_intervals``. Each result must have one value per
                detected breath, in the same order. The results are m3resp
                ``IntervalData``, or eitprocessing results with ``values``
                (pixel results with one row-column grid per breath). Units
                are taken from each result.
            stored_breaths: Existing breaths to reuse when modality and exact
                start and end times match. ``None`` uses newly converted breaths.

        Returns:
            list[IntervalData]: Available results in TIV, EELI, pixel-TIV,
                pixel-EELI order, sharing their breath objects. Pixel values
                are ``PixelMap`` objects with row-column grids. An empty list
                is returned when all four results are absent.

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
                ("pixel_eeli", "eitprocessing.EELI", True),
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
        for result, method, as_pixel_maps in per_breath:
            if isinstance(result, IntervalData):
                # Results of m3resp: the values stay, the breaths are the
                # stored ones.
                if len(result) != len(breaths):
                    raise ValueError(
                        f"Got {len(result)} per-breath values but {len(breaths)} "
                        "breaths; the breaths must be the ones the values were "
                        "computed over."
                    )
                results.append(
                    dataclasses.replace(
                        result,
                        intervals=list(breaths),
                        metadata=dict(result.metadata),
                    )
                )
                continue
            results.append(
                sparse_data_to_interval_data(
                    result,
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
