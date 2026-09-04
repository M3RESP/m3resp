"""Native fallback implementations of `ReSurfEMGAdapter`, used when the optional `resurfemg` package is unavailable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.processing.filters import harmonic_notch_filter
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
    onoff_from_slope,
)
from m3resp.processing.metrics import (
    amplitude_at_peaks,
    area_under_baseline,
    pseudo_slope,
    respiratory_rate_from_indices,
    time_to_peak,
    window_integral,
)
from m3resp.processing.peaks import (
    detect_emg_breath_peaks,
    detect_occluded_breath_peaks,
    detect_ventilator_breath_peaks,
)
from m3resp.processing.ventilator import estimate_peep
from m3resp.processing.windows import rolling_envelope

from ._protocols import _PostprocessingOpsProtocol
from ._shared import (
    _category_for_function,
    _mask_invalid,
    _missing_postprocessing_dependency,
    _normalize_selected_postprocessing,
    _require_emg_recording,
    _unavailable_postprocessing_result,
)
from ._signals import peak_indices_from_events, ventilator_signals


class _DefaultsMixin:
    def _preprocess_default(
        self,
        recording: Any,
        *,
        channel: int = 0,
        high_pass_hz: float = 20.0,
        low_pass_hz: float | None = None,
        envelope_window_seconds: float = 0.25,
        envelope_method: str = "rms",
        notch_base_frequency: float | None = None,
        notch_max_frequency: float | None = None,
        notch_quality_factor: float = 30.0,
    ) -> dict[str, Any]:
        """Run the Stage 1 EMG preprocessing pipeline through ReSurfEMG.

        The band-pass defaults to 20-500 Hz, the range respiratory-sEMG
        literature specifies. The high-pass is deliberately *not* set low
        enough to double as ECG suppression: removing ECG is the job of a
        dedicated gating step (``emg.ecg_gating``, which the ``"emg"`` preset
        runs by default), because a high-pass steep enough to attenuate the
        QRS complex still leaves its higher-frequency content inside the pass
        band.

        ``envelope_method`` selects the envelope computed on the band-passed
        signal - ``"rms"`` (default) or ``"arv"``. RMS is what the literature
        specifies; ARV is kept as an explicit opt-in because it is not an RMS
        equivalent on real bursty sEMG. The choice is recorded in the returned
        ``"filter"`` mapping so a later envelope recomputation (e.g. after ECG
        gating) reuses the same method rather than silently switching.

        ``notch_base_frequency`` opts into harmonic notch filtering (e.g.
        ``50.0`` for mains hum, or a co-recorded EIT device's frame rate, which
        injects a harmonic comb into the sEMG whenever the EIT device is
        running simultaneously). It is applied to the band-passed signal, after
        ``emg_bandpass_butter`` and before the envelope is computed, so a
        narrow high-pass alone (which only removes the fundamental) doesn't
        leave higher harmonics inside the pass band untouched.
        """

        try:
            import numpy as np
            from resurfemg.preprocessing.filtering import emg_bandpass_butter
        except ImportError as exc:
            raise OptionalDependencyError(
                "EMG preprocessing requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"`.'
            ) from exc

        _require_emg_recording(recording)

        metadata = dict(recording["metadata"])
        fs = float(metadata["fs"])
        array = recording["array"]
        raw = np.asarray(array[channel], dtype=float)

        if low_pass_hz is None:
            low_pass_hz = min(fs / 2 * 0.95, 500)

        filtered = emg_bandpass_butter(
            emg_raw=raw,
            high_pass=high_pass_hz,
            low_pass=low_pass_hz,
            fs_emg=fs,
        )
        if notch_base_frequency is not None:
            # Default the notch's reach to Nyquist, not `low_pass_hz`: a
            # harmonic landing at or just past the low-pass cutoff (e.g. the
            # EIT frame-rate comb's 10th harmonic sitting on a 500 Hz
            # low-pass edge) is only partially attenuated by the low-pass
            # filter's finite roll-off, so it must still be fully inside the
            # notch's stopband rather than at its boundary.
            filtered = harmonic_notch_filter(
                filtered,
                base_frequency=notch_base_frequency,
                sample_frequency=fs,
                max_frequency=notch_max_frequency or (fs / 2),
                quality_factor=notch_quality_factor,
            )
        envelope_window_samples = max(1, int(envelope_window_seconds * fs))
        envelope = rolling_envelope(
            filtered,
            window_length=envelope_window_samples,
            method=envelope_method,
        )

        return {
            **recording,
            "channel": channel,
            "fs": fs,
            "raw_channel": raw,
            "filtered": filtered,
            "envelope": envelope,
            "filter": {
                "high_pass_hz": high_pass_hz,
                "low_pass_hz": low_pass_hz,
                "envelope_window_seconds": envelope_window_seconds,
                "envelope_method": envelope_method,
                "notch_base_frequency": notch_base_frequency,
                "notch_max_frequency": (
                    (notch_max_frequency or (fs / 2))
                    if notch_base_frequency is not None
                    else None
                ),
                "notch_quality_factor": (
                    notch_quality_factor if notch_base_frequency is not None else None
                ),
            },
        }

    def _detect_breaths_default(
        self,
        processed_emg: Any,
        *,
        min_breath_width_seconds: float = 1.0,
        half_window_seconds: float = 0.5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run ReSurfEMG EMG breath detection and return common rows."""

        if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
            raise UnsupportedWorkflowError(
                "Default EMG breath detection expects processed EMG data from "
                "`preprocess_emg()`. Pass `detector=callable` to normalize "
                "custom detections."
            )

        fs = float(processed_emg["fs"])
        envelope = processed_emg["envelope"]
        min_width_samples = max(1, int(min_breath_width_seconds * fs))
        half_window_samples = max(1, int(half_window_seconds * fs))

        peak_indices = detect_emg_breath_peaks(
            envelope,
            min_peak_width_samples=min_width_samples,
            **kwargs,
        )

        events = []
        for peak_index in peak_indices:
            start_index = max(0, int(peak_index) - half_window_samples)
            end_index = min(len(envelope) - 1, int(peak_index) + half_window_samples)
            events.append(
                {
                    "start_time": start_index / fs,
                    "end_time": end_index / fs,
                    "peak_time": int(peak_index) / fs,
                    "start_index": start_index,
                    "peak_index": int(peak_index),
                    "end_index": end_index,
                    "sample_frequency": fs,
                    "signal_name": processed_emg["channel"],
                    "source": "resurfemg.detect_emg_breaths",
                }
            )

        return events

    def _postprocess_default(
        self: _PostprocessingOpsProtocol,
        processed_emg: Any,
        events: Sequence[BreathEvent] | None = None,
        *,
        ventilator: Any | None = None,
        ventilator_pressure_channel: int = 0,
        ventilator_flow_channel: int = 1,
        ventilator_volume_channel: int = 2,
        ventilator_fs: float | None = None,
        ventilator_breath_width_seconds: float = 0.5,
        peep: float | None = None,
        baseline_window_seconds: float = 30.0,
        baseline_step_seconds: float = 1.0,
        baseline_percentile: float = 33.0,
        slope_window_seconds: float = 0.5,
        aub_window_seconds: float = 5.0,
        selected_functions: dict[str, dict[str, bool]] | None = None,
    ) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise OptionalDependencyError("EMG postprocessing requires numpy.") from exc

        if not isinstance(processed_emg, dict) or "envelope" not in processed_emg:
            raise UnsupportedWorkflowError(
                "Default EMG postprocessing expects processed EMG data from "
                "`preprocess_emg()`."
            )

        envelope = np.asarray(processed_emg["envelope"], dtype=float)
        fs = float(processed_emg["fs"])
        window_samples = max(1, int(baseline_window_seconds * fs))
        step_samples = max(1, int(baseline_step_seconds * fs))
        selected = _normalize_selected_postprocessing(selected_functions)
        enabled = selected.__contains__

        computed: dict[str, Any] = {
            "baseline": {},
            "event_detection": {},
            "features": {},
            "quality_assessment": {},
        }
        skipped: dict[str, str] = {}

        peak_indices = peak_indices_from_events(events, fs)
        peak_indices_array = np.asarray(peak_indices, dtype=int)
        unavailable_reason = _missing_postprocessing_dependency()
        if unavailable_reason is not None:
            return _unavailable_postprocessing_result(
                selected=selected,
                peak_indices=peak_indices_array,
                computed=computed,
                reason=unavailable_reason,
                settings={
                    "baseline_window_seconds": baseline_window_seconds,
                    "baseline_step_seconds": baseline_step_seconds,
                    "baseline_percentile": baseline_percentile,
                    "slope_window_seconds": slope_window_seconds,
                    "aub_window_seconds": aub_window_seconds,
                    "ventilator_breath_width_seconds": ventilator_breath_width_seconds,
                    "peep": peep,
                },
            )

        baseline = None
        if enabled(("baseline", "moving_baseline")):
            moving_baseline = self.moving_baseline(
                envelope,
                window_samples=window_samples,
                step_samples=step_samples,
                percentile=baseline_percentile,
            )
            computed["baseline"]["moving_baseline"] = moving_baseline
            baseline = moving_baseline

        if enabled(("baseline", "slopesum_baseline")):
            slopesum_baseline = self.slopesum_baseline(
                envelope,
                window_samples=window_samples,
                step_samples=step_samples,
                sample_frequency=fs,
                percentile=baseline_percentile,
                moving_average_samples=max(1, int(fs // 2)),
                percentile_window_samples=max(1, int(fs)),
            )
            computed["baseline"]["slopesum_baseline"] = {
                "baseline": slopesum_baseline[0],
                "running_mean": slopesum_baseline[1],
                "running_std": slopesum_baseline[2],
                "series": slopesum_baseline[3],
            }
            baseline = slopesum_baseline[0]

        vent_signals = ventilator_signals(
            ventilator,
            pressure_channel=ventilator_pressure_channel,
            flow_channel=ventilator_flow_channel,
            volume_channel=ventilator_volume_channel,
            fs=ventilator_fs,
        )

        ventilator_breath_indices = np.asarray([], dtype=int)
        if vent_signals is not None:
            v_vent = vent_signals["volume"]
            p_vent = vent_signals["pressure"]
            vent_fs = float(vent_signals["fs"])
            vent_width_samples = max(1, int(ventilator_breath_width_seconds * vent_fs))
            if enabled(("event_detection", "detect_ventilator_breath")):
                ventilator_breath_indices = np.asarray(
                    detect_ventilator_breath_peaks(
                        v_vent,
                        start_index=0,
                        end_index=len(v_vent) - 1,
                        width_samples=vent_width_samples,
                    ),
                    dtype=int,
                )
                computed["event_detection"]["detect_ventilator_breath"] = (
                    ventilator_breath_indices
                )

            pocc_indices = np.asarray([], dtype=int)
            if enabled(("event_detection", "find_occluded_breaths")):
                if peep is None:
                    peep = estimate_peep(p_vent, v_vent)
                pocc_indices = np.asarray(
                    detect_occluded_breath_peaks(
                        p_vent,
                        sample_frequency=vent_fs,
                        peep=peep,
                    ),
                    dtype=int,
                )
                computed["event_detection"]["find_occluded_breaths"] = pocc_indices

            if enabled(("quality_assessment", "detect_non_consecutive_manoeuvres")):
                if len(ventilator_breath_indices) and len(pocc_indices):
                    computed["quality_assessment"][
                        "detect_non_consecutive_manoeuvres"
                    ] = self.detect_non_consecutive_manoeuvres(
                        ventilator_breath_indices,
                        pocc_indices,
                    )
                else:
                    skipped["quality_assessment.detect_non_consecutive_manoeuvres"] = (
                        "Needs ventilator breath and manoeuvre indices."
                    )

            if (
                enabled(("quality_assessment", "evaluate_respiratory_rates"))
                and len(ventilator_breath_indices) >= 2
            ):
                computed["quality_assessment"]["ventilator_respiratory_rate"] = (
                    self.run_postprocessing_function(
                        "features",
                        "respiratory_rate",
                        ventilator_breath_indices,
                        vent_fs,
                    )
                )
            else:
                skipped["quality_assessment.ventilator_respiratory_rate"] = (
                    "Needs at least two ventilator breaths."
                )
        else:
            skipped.update(
                {
                    "event_detection.find_occluded_breaths": "Needs ventilator pressure and PEEP inputs.",
                    "event_detection.detect_ventilator_breath": "Needs ventilator volume input.",
                    "quality_assessment.detect_non_consecutive_manoeuvres": "Needs ventilator breath and manoeuvre indices.",
                    "quality_assessment.ventilator_respiratory_rate": "Needs ventilator volume input.",
                }
            )

        start_indices = None
        end_indices = None
        start_end_validity = None
        if len(peak_indices_array) and baseline is not None:
            if enabled(("event_detection", "onoffpeak_baseline_crossing")):
                computed["event_detection"]["onoffpeak_baseline_crossing"] = (
                    onoff_from_baseline_crossings(
                        envelope,
                        baseline,
                        peak_indices_array,
                    )
                )
                start_indices, end_indices, _valid_starts, _valid_ends, valid_peaks = (
                    computed["event_detection"]["onoffpeak_baseline_crossing"]
                )
                start_end_validity = np.asarray(valid_peaks, dtype=bool)
            slope_window_samples = max(1, int(slope_window_seconds * fs))
            if enabled(("event_detection", "onoffpeak_slope_extrapolation")):
                computed["event_detection"]["onoffpeak_slope_extrapolation"] = (
                    onoff_from_slope(
                        envelope,
                        sample_frequency=fs,
                        peak_indices=peak_indices_array,
                        slope_window=slope_window_samples,
                    )
                )

            if start_indices is not None and end_indices is not None:
                if enabled(("features", "time_to_peak")):
                    absolute_times, percent_times = time_to_peak(
                        envelope,
                        start_indices,
                        end_indices,
                    )
                    computed["features"]["time_to_peak"] = (
                        _mask_invalid(absolute_times, start_end_validity),
                        _mask_invalid(percent_times, start_end_validity),
                    )
                if enabled(("features", "pseudo_slope")):
                    computed["features"]["pseudo_slope"] = _mask_invalid(
                        pseudo_slope(
                            envelope,
                            start_indices,
                            end_indices,
                        ),
                        start_end_validity,
                    )
                if enabled(("features", "amplitude")):
                    computed["features"]["amplitude"] = amplitude_at_peaks(
                        envelope,
                        peak_indices_array,
                        baseline,
                    )
                if enabled(("features", "time_product")):
                    computed["features"]["time_product"] = _mask_invalid(
                        window_integral(
                            envelope,
                            fs,
                            start_indices,
                            end_indices,
                            baseline,
                        ),
                        start_end_validity,
                    )
                if enabled(("features", "area_under_baseline")):
                    areas, references = area_under_baseline(
                        envelope,
                        fs,
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        max(1, int(aub_window_seconds * fs)),
                        baseline,
                    )
                    computed["features"]["area_under_baseline"] = (
                        _mask_invalid(areas, start_end_validity),
                        _mask_invalid(references, start_end_validity),
                    )
            else:
                skipped["features"] = (
                    "Needs event_detection.onoffpeak_baseline_crossing."
                )

            if (
                enabled(("features", "respiratory_rate"))
                and len(peak_indices_array) >= 2
            ):
                computed["features"]["respiratory_rate"] = (
                    respiratory_rate_from_indices(peak_indices_array, fs)
                )
            elif enabled(("features", "respiratory_rate")):
                skipped["features.respiratory_rate"] = "Needs at least two EMG breaths."

            if enabled(("quality_assessment", "snr_pseudo")):
                computed["quality_assessment"]["snr_pseudo"] = self.snr_pseudo(
                    envelope,
                    peak_indices_array,
                    baseline,
                    sample_frequency=fs,
                )
            if (
                enabled(("quality_assessment", "percentage_under_baseline"))
                and start_indices is not None
                and end_indices is not None
            ):
                computed["quality_assessment"]["percentage_under_baseline"] = (
                    self.percentage_under_baseline(
                        envelope,
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        baseline,
                        sample_frequency=fs,
                    )
                )
            if (
                enabled(("quality_assessment", "detect_local_high_aub"))
                and "area_under_baseline" in computed["features"]
            ):
                aubs = computed["features"]["area_under_baseline"][0]
                computed["quality_assessment"]["detect_local_high_aub"] = (
                    self.detect_local_high_aub(aubs)
                )
            if (
                enabled(("quality_assessment", "detect_extreme_time_products"))
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["detect_extreme_time_products"] = (
                    self.detect_extreme_time_products(time_products)
                )
            if (
                enabled(("quality_assessment", "evaluate_bell_curve_error"))
                and start_indices is not None
                and end_indices is not None
                and "time_product" in computed["features"]
            ):
                time_products = computed["features"]["time_product"]
                computed["quality_assessment"]["evaluate_bell_curve_error"] = (
                    self.evaluate_bell_curve_error(
                        peak_indices_array,
                        start_indices,
                        end_indices,
                        envelope,
                        time_products,
                        sample_frequency=fs,
                    )
                )
            if (
                enabled(("quality_assessment", "evaluate_event_timing"))
                and vent_signals is not None
                and len(ventilator_breath_indices)
            ):
                paired_count = min(
                    len(peak_indices_array), len(ventilator_breath_indices)
                )
                computed["quality_assessment"]["evaluate_event_timing"] = (
                    self.evaluate_event_timing(
                        peak_indices_array[:paired_count] / fs,
                        ventilator_breath_indices[:paired_count]
                        / float(vent_signals["fs"]),
                    )
                )
            elif enabled(("quality_assessment", "evaluate_event_timing")):
                skipped["quality_assessment.evaluate_event_timing"] = (
                    "Needs ventilator breath timing."
                )

            # Deliberately independent of 'evaluate_event_timing' above: its
            # only real prerequisite is that 'ventilator_respiratory_rate' was
            # computed (near the top of this function), which does not
            # require 'evaluate_event_timing' to be selected. Nesting this
            # under that block previously meant selecting
            # 'evaluate_respiratory_rates' alone (without also selecting
            # 'evaluate_event_timing') silently produced nothing - no result,
            # no skip reason.
            if (
                enabled(("quality_assessment", "evaluate_respiratory_rates"))
                and "ventilator_respiratory_rate" in computed["quality_assessment"]
            ):
                rr_vent = computed["quality_assessment"]["ventilator_respiratory_rate"][
                    0
                ]
                computed["quality_assessment"]["evaluate_respiratory_rates"] = (
                    self.evaluate_respiratory_rates(
                        peak_indices_array,
                        len(envelope) / fs,
                        rr_vent,
                    )
                )
            elif enabled(("quality_assessment", "evaluate_respiratory_rates")):
                skipped["quality_assessment.evaluate_respiratory_rates"] = (
                    "Needs ventilator respiratory rate (requires "
                    "'evaluate_respiratory_rates' selected together with at "
                    "least two ventilator breaths)."
                )
        else:
            for name in (
                "onoffpeak_baseline_crossing",
                "onoffpeak_slope_extrapolation",
                "time_to_peak",
                "pseudo_slope",
                "amplitude",
                "time_product",
                "area_under_baseline",
                "respiratory_rate",
                "snr_pseudo",
                "percentage_under_baseline",
                "detect_local_high_aub",
                "detect_extreme_time_products",
                "evaluate_bell_curve_error",
            ):
                category = _category_for_function(name)
                if category is not None and enabled((category, name)):
                    skipped[f"{category}.{name}"] = (
                        "Needs detected EMG breath peaks and baseline."
                    )
            if enabled(("quality_assessment", "evaluate_event_timing")):
                skipped["quality_assessment.evaluate_event_timing"] = (
                    "Needs detected EMG breath peaks."
                )
            if enabled(("quality_assessment", "evaluate_respiratory_rates")):
                skipped["quality_assessment.evaluate_respiratory_rates"] = (
                    "Needs detected EMG breath peaks."
                )

        if enabled(("quality_assessment", "pocc_quality")):
            skipped["quality_assessment.pocc_quality"] = "Needs Pocc ventilator inputs."
        if enabled(("quality_assessment", "interpeak_dist")):
            skipped["quality_assessment.interpeak_dist"] = "Needs ECG peak indices."

        return {
            "available": self.available_postprocessing(),
            "computed": computed,
            "skipped": skipped,
            "peak_indices": peak_indices_array,
            "settings": {
                "baseline_window_seconds": baseline_window_seconds,
                "baseline_step_seconds": baseline_step_seconds,
                "baseline_percentile": baseline_percentile,
                "slope_window_seconds": slope_window_seconds,
                "aub_window_seconds": aub_window_seconds,
                "ventilator_breath_width_seconds": ventilator_breath_width_seconds,
                "peep": peep,
            },
        }
