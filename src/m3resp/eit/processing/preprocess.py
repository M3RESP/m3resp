"""Preprocess a loaded EIT recording in one call.

``preprocess_eit_recording`` runs the EIT calculations of
``m3resp.eit.processing`` in a fixed order: the breathing rate and the heart
rate, a filter of the images (MDN, Butterworth low-pass or band-pass, or
none), the global impedance of the filtered images, the breaths, and the TIV
and the EELI of each breath, for the whole waveform and for each pixel.
``M3Session.preprocess_eit`` calls it.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from m3resp.data import Signal
from m3resp.eit.processing.breath_detection import (
    BreathDetectionSettings,
    find_breaths,
)
from m3resp.eit.processing.breath_inputs import breaths_to_interval_data
from m3resp.eit.processing.butterworth import butterworth_filter_images
from m3resp.eit.processing.eeli import (
    end_expiratory_lung_impedance,
    pixel_end_expiratory_lung_impedance,
)
from m3resp.eit.processing.impedance import global_impedance
from m3resp.eit.processing.mdn_filter import mdn_filter
from m3resp.eit.processing.rate_detection import RateDetectionSettings, detect_rates
from m3resp.eit.processing.tidal_impedance_variation import (
    pixel_tidal_impedance_variation,
    tidal_impedance_variation,
)

#: The filters of ``preprocess_eit_recording``.
FILTER_MODES = ("mdn", "lowpass", "bandpass", "none")

#: The keys of the per-breath results, in the order they are stored.
PER_BREATH_KEYS = ("continuous_tiv", "eeli", "pixel_tiv", "pixel_eeli")


class _Recording(Protocol):
    """The parts of a loaded EIT recording that preprocessing reads."""

    @property
    def pixel_impedance(self) -> Signal: ...

    @property
    def raw_global_impedance(self) -> Signal: ...


def preprocess_eit_recording(
    recording: _Recording,
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
    """Filter the EIT images of a recording and compute the breath results.

    The MDN filter needs the breathing rate and the heart rate, so the rates
    are found for ``filter_mode="mdn"`` even when ``compute_rates`` is False.
    The breaths are found on the global impedance of the filtered images.
    Without a filter, this is the raw global impedance of the recording (the
    waveform of the file for Draeger and Timpel, the sum of the pixels for
    Sentec).

    Args:
        recording (_Recording): The loaded recording, an ``EITRecording``
            (``session.eit``), with the raw images in ``pixel_impedance`` and
            the raw global impedance in ``raw_global_impedance``.
        subject_type (Literal["adult", "neonate"]): Sets the rate ranges of
            the rate detection.
        welch_window_seconds (float): Length of one Welch window of the rate
            detection, in seconds.
        filter_mode (str): ``"mdn"``, ``"lowpass"``, ``"bandpass"`` or
            ``"none"`` (upper or lower case).
        filter_enabled (bool): False gives ``filter_mode="none"``.
        lowpass_hz (float): Low-pass cutoff, or upper band-pass cutoff, in Hz.
        highpass_hz (float): Lower band-pass cutoff, in Hz.
        filter_order (int): Butterworth order of the low-pass and band-pass
            filters, 1 to 10.
        breath_min_duration_seconds (float): Shortest expected breath, in
            seconds.
        compute_rates (bool): Find the breathing rate and the heart rate.
        compute_breath_intervals (bool): Find the breaths.
        compute_continuous_tiv (bool): Compute the TIV of each breath.
        compute_eeli (bool): Compute the EELI of each breath.
        compute_pixel_tiv (bool): Compute one map of TIV per breath.
        compute_pixel_eeli (bool): Compute one map of EELI per breath.
        include_filtered_data (bool): Return the filtered images under
            ``filtered_eit``.
        include_global_impedance (bool): Return the raw global impedance and
            the global impedance of the filtered images. The breath
            calculations use the global impedance also when this is False.

    Returns:
        dict[str, Any]: ``raw_eit`` (the raw images, a ``Signal``),
            ``raw_global_impedance`` and ``filtered_global_impedance``
            (``Signal`` objects, times in seconds), ``filtered_eit`` (the filtered
            images, with the label of the filter in ``metadata["label"]``),
            ``filter_mode``, ``filter_captures`` (the in-between results of
            the filter), ``rate_detector`` (``RateDetectionSettings``),
            ``rate_captures`` (the spectra), ``respiratory_rate_hz`` and
            ``heart_rate_hz`` (in Hz), ``breath_intervals`` (an
            ``IntervalData`` of ``BreathEvent`` objects), ``continuous_tiv`` and
            ``eeli`` (one value per breath), and ``pixel_tiv`` and
            ``pixel_eeli`` (one ``PixelMap`` per breath, with the timing of
            the global breaths). A result that is switched off is None; the
            captures of a skipped calculation are empty.

    Raises:
        ValueError: If the filter mode is unknown, if the Butterworth order is
            outside 1 to 10, or if TIV, EELI, pixel TIV or pixel EELI is
            asked for without the breaths.
    """

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
    mode = "none" if not filter_enabled else filter_mode.lower()
    if mode not in FILTER_MODES:
        raise ValueError(
            "filter_mode must be one of: 'mdn', 'lowpass', 'bandpass', 'none'."
        )

    raw_eit = recording.pixel_impedance
    raw_global_impedance = (
        recording.raw_global_impedance if include_global_impedance else None
    )

    rate_detector = None
    rate_captures: dict[str, Any] = {}
    respiratory_rate_hz = None
    heart_rate_hz = None
    if compute_rates or mode == "mdn":
        rate_detector = RateDetectionSettings(
            subject_type, welch_window=welch_window_seconds
        )
        respiratory_rate_hz, heart_rate_hz = detect_rates(
            raw_eit,
            rate_detector,
            captures=rate_captures,
            suppress_length_warnings=True,
            suppress_edge_case_warning=True,
        )

    filtered_eit, filter_captures = _filter_images(
        raw_eit,
        mode,
        respiratory_rate_hz=respiratory_rate_hz,
        heart_rate_hz=heart_rate_hz,
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
        order=filter_order,
    )

    filtered_global_impedance = raw_global_impedance
    if filtered_eit is not raw_eit and include_global_impedance:
        filtered_global_impedance = _global_impedance_of(filtered_eit)

    breath_intervals = None
    continuous_tiv = None
    eeli = None
    pixel_tiv = None
    pixel_eeli = None
    if compute_breath_intervals:
        if filtered_global_impedance is None:
            filtered_global_impedance = (
                recording.raw_global_impedance
                if filtered_eit is raw_eit
                else _global_impedance_of(filtered_eit)
            )
        breaths = find_breaths(
            filtered_global_impedance,
            BreathDetectionSettings(minimum_duration=breath_min_duration_seconds),
        )
        breath_intervals = breaths_to_interval_data(breaths)
        if compute_continuous_tiv:
            continuous_tiv = tidal_impedance_variation(
                filtered_global_impedance, breaths, name="continuous_tivs"
            )
        if compute_eeli:
            eeli = end_expiratory_lung_impedance(
                filtered_global_impedance, breaths, name="continuous_eelis"
            )
        if compute_pixel_tiv:
            pixel_tiv = pixel_tidal_impedance_variation(
                filtered_eit, breaths, name="pixel_tivs"
            )
        if compute_pixel_eeli:
            pixel_eeli = pixel_end_expiratory_lung_impedance(
                filtered_eit, breaths, name="pixel_eelis"
            )

    return {
        "raw_eit": raw_eit,
        "raw_global_impedance": raw_global_impedance,
        "filtered_eit": filtered_eit if include_filtered_data else None,
        "filtered_global_impedance": filtered_global_impedance,
        "filter_mode": mode,
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


def _filter_images(
    images: Signal,
    mode: str,
    *,
    respiratory_rate_hz: float | None,
    heart_rate_hz: float | None,
    lowpass_hz: float,
    highpass_hz: float,
    order: int,
) -> tuple[Signal, dict[str, Any]]:
    """Return the images after the filter of ``mode``, and the captures.

    Args:
        images (Signal): The raw images.
        mode (str): ``"mdn"``, ``"lowpass"``, ``"bandpass"`` or ``"none"``.
        respiratory_rate_hz (float | None): Breathing rate in Hz, for MDN.
        heart_rate_hz (float | None): Heart rate in Hz, for MDN.
        lowpass_hz (float): Low-pass cutoff, or upper band-pass cutoff, in Hz.
        highpass_hz (float): Lower band-pass cutoff, in Hz.
        order (int): Butterworth order of the low-pass and band-pass filters.

    Returns:
        tuple[Signal, dict[str, Any]]: The filtered images, with
            ``metadata["label"]`` set to ``"<mode>_filtered"``, and the
            captures of the filter. For ``"none"``, ``images`` itself and an
            empty dict.
    """

    captures: dict[str, Any] = {}
    if mode == "mdn":
        assert respiratory_rate_hz is not None
        assert heart_rate_hz is not None
        filtered = mdn_filter(
            images,
            respiratory_rate=respiratory_rate_hz,
            heart_rate=heart_rate_hz,
            name="MDN-filtered EIT data",
            captures=captures,
        )
    elif mode in ("lowpass", "bandpass"):
        filtered, captures = butterworth_filter_images(
            images,
            mode=cast(Literal["lowpass", "bandpass"], mode),
            cutoff_frequency=lowpass_hz
            if mode == "lowpass"
            else (highpass_hz, lowpass_hz),
            order=order,
            operation=f"preprocess(filter_mode={mode!r})",
        )
    else:
        return images, captures
    filtered.metadata["label"] = f"{mode}_filtered"
    return filtered, captures


def _global_impedance_of(filtered_eit: Signal) -> Signal:
    """Return the global impedance of filtered images.

    Args:
        filtered_eit (Signal): Filtered images with their label in
            ``metadata["label"]``.

    Returns:
        Signal: The sum of the pixels of each image, named
            ``"Global impedance (<label>)"``.
    """

    label = filtered_eit.metadata["label"]
    return global_impedance(filtered_eit, name=f"Global impedance ({label})")
