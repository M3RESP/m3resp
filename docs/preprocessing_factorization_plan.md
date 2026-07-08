# Preprocessing Factorization Plan

This plan describes how to move overlapping `eitprocessing` and `resurfemg`
functionality into `m3resp` while preserving existing behavior. The goal is
not to merge the two scientific packages into one algorithm. The goal is to
extract the reusable signal-processing, event, metric, and provenance
primitives that both packages already need, then keep modality-specific EIT
and EMG wrappers on top.

## Recommendation

Use a layered migration:

1. Add shared primitives in `m3resp`.
2. Refactor adapters to call those primitives only where behavior can be kept
   byte-for-byte or numerically equivalent.
3. Keep EIT-specific image, pixel, ROI, TIV, and EELI logic separate.
4. Keep EMG-specific ECG removal, gating, envelope defaults, and Pocc/ventilator
   conventions separate.
5. Protect each extraction with regression tests against the current
   `eitprocessing` and `resurfemg` calls.

The current adapter boundary should remain the public migration path:

- `m3resp.adapters.eitprocessing_adapter.EITProcessingAdapter`
- `m3resp.adapters.resurfemg_adapter.ReSurfEMGAdapter`
- `m3resp.data.Signal`
- `m3resp.core.events.BreathEvent`
- `m3resp.data.ParameterResult`
- `m3resp.data.QualityFlag`

## Current Function Map

### EIT Processing

| Area | Current functions/classes | Notes |
|---|---|---|
| Loading | `load_eit_data`, Draeger/Sentec/Timpel loaders | Keep modality/vendor-specific. |
| Containers | `ContinuousData`, `EITData`, `Sequence`, `IntervalData`, `SparseData`, `Breath` | Convert at adapter boundary into `m3resp` runtime objects. |
| Filters | `ButterworthFilter`, `LowPassFilter`, `HighPassFilter`, `BandPassFilter`, `BandStopFilter`, `MDNFilter` | Butterworth and notch primitives are reusable. MDN policy remains EIT-specific. |
| Windowing | `MovingAverage` | Reusable. |
| Detection | `BreathDetection`, `PixelBreath`, `RateDetection` | Share low-level peak/window primitives only. Keep EIT breath semantics separate. |
| Parameters | `TIV`, `EELI` | Keep EIT-specific. Reuse generic per-breath window metric helpers where possible. |
| ROI/maps | `PixelMask`, `PixelMaskCollection`, `TIVLungspace`, `AmplitudeLungspace`, `WatershedLungspace`, `FilterROIBySize` | Keep EIT-specific. |

### ReSurfEMG Processing

| Area | Current functions/classes | Notes |
|---|---|---|
| Filters | `emg_bandpass_butter`, `emg_lowpass_butter`, `emg_highpass_butter`, `notch_filter`, `compute_power_loss` | Butterworth, notch, and power-loss helpers are reusable. |
| Envelopes | `full_rolling_rms`, `naive_rolling_rms`, `full_rolling_arv`, `rolling_rms_ci`, `rolling_arv_ci` | Reusable rolling-window primitives with EMG wrappers. |
| ECG removal | `detect_ecg_peaks`, `gating`, `wavelet_denoising` | Mostly EMG-specific. Can reuse peak detection and masking/window helpers. |
| Baseline | `moving_baseline`, `slopesum_baseline` | Reusable shape, but preserve EMG defaults. |
| Event detection | `find_occluded_breaths`, `onoffpeak_baseline_crossing`, `onoffpeak_slope_extrapolation`, `detect_ventilator_breath`, `detect_emg_breaths`, `find_linked_peaks` | Share peak, crossing, interval, and closest-event helpers. |
| Features | `time_to_peak`, `pseudo_slope`, `amplitude`, `time_product`, `area_under_baseline`, `respiratory_rate` | Strong candidate for shared per-breath metric helpers. |
| Quality | `snr_pseudo`, `pocc_quality`, `interpeak_dist`, `percentage_under_baseline`, `detect_local_high_aub`, `detect_extreme_time_products`, `detect_non_consecutive_manoeuvres`, `evaluate_bell_curve_error`, `evaluate_event_timing`, `evaluate_respiratory_rates` | Map results into `QualityFlag`; keep clinical thresholds configurable. |

## Target Package Layout

Add shared modules under `src/m3resp/processing/`:

```text
src/m3resp/processing/
|-- __init__.py
|-- filters.py          Butterworth, notch, harmonic notch, power loss
|-- windows.py          moving average, rolling RMS/ARV, rolling CI helpers
|-- peaks.py            find_peaks wrapper, peak/valley pairing, peak linking
|-- intervals.py        sample/time interval conversion and baseline crossings
|-- metrics.py          amplitude, time-to-peak, integral/time product, RR
|-- quality.py          generic threshold/pass-fail helpers
`-- provenance.py       shared ProcessingStep metadata helpers, if needed
```

Keep modality policies in dedicated modules:

```text
src/m3resp/modalities/
|-- eit.py              EIT session-facing helpers
|-- emg.py              EMG session-facing helpers
`-- ventilator.py       Add only if ventilator helpers outgrow EMG postprocessing
```

Keep package compatibility in adapters:

```text
src/m3resp/adapters/
|-- eitprocessing_adapter.py
`-- resurfemg_adapter.py
```

## Implementation Phases

### Phase 1: Shared Filters

Add `m3resp.processing.filters` with:

- `butterworth_filter(values, *, filter_type, cutoff_frequency, sample_frequency, order, axis=-1)`
- `lowpass_filter(...)`
- `highpass_filter(...)`
- `bandpass_filter(...)`
- `bandstop_filter(...)`
- `notch_filter(values, *, frequency, sample_frequency, quality_factor)`
- `harmonic_notch_filter(values, *, base_frequency, sample_frequency, max_frequency=None, distance=None)`
- `compute_power_loss(original, processed, *, original_frequency, processed_frequency, n_segment=None, percent_overlap=25)`

Use `scipy.signal.butter(..., output="sos")` and `scipy.signal.sosfiltfilt`
for Butterworth filters, matching both packages' current implementation.

Refactor targets:

- `resurfemg.preprocessing.filtering.emg_*_butter`
- `eitprocessing.filters.butterworth_filters.ButterworthFilter.apply`
- EIT adapter lowpass/bandpass path

Tests:

- Compare output against current ReSurfEMG functions for synthetic EMG arrays.
- Compare output against `eitprocessing.ButterworthFilter.apply` for 1D and
  EIT-shaped arrays with `axis=0`.
- Confirm NaN behavior is explicit and documented.

### Phase 2: Shared Windows and Envelopes

Add `m3resp.processing.windows` with:

- `moving_average(values, *, window_size, window_function=None, padding_type="edge")`
- `rolling_rms(values, *, window_length, center=True, min_periods=1)`
- `rolling_arv(values, *, window_length, center=True, min_periods=1)`
- `rolling_rms_ci(values, *, window_length, alpha=0.05)`
- `rolling_arv_ci(values, *, window_length, alpha=0.05)`

Refactor targets:

- `eitprocessing.features.moving_average.MovingAverage`
- `resurfemg.preprocessing.envelope.*`
- EMG default preprocessing in `ReSurfEMGAdapter._preprocess_default`

Tests:

- Exact comparison to `MovingAverage.apply`.
- Exact or tolerance comparison to ReSurfEMG RMS/ARV outputs.
- Boundary behavior for short windows, even windows, and window length longer
  than the signal.

### Phase 3: Shared Peak and Interval Primitives

Add `m3resp.processing.peaks` and `m3resp.processing.intervals` with:

- `detect_peaks(values, *, height=None, prominence=None, width=None, distance=None, invert=False)`
- `detect_peaks_above_moving_average(values, moving_average, *, minimum_distance)`
- `pair_valley_peak_valley(values, peak_indices, valley_indices)`
- `remove_duplicate_extrema(values, peak_indices, valley_indices)`
- `remove_low_amplitude_peaks(values, peak_indices, valley_indices, fraction)`
- `baseline_crossings(values, baseline)`
- `onoff_from_baseline_crossings(values, baseline, peak_indices)`
- `onoff_from_slope(values, *, sample_frequency, peak_indices, slope_window)`
- `closest_event_indices(reference_times, candidate_times)`
- `sample_intervals_to_breath_events(...)`

Refactor targets:

- EIT `BreathDetection` internals
- ReSurfEMG `detect_emg_breaths`
- ReSurfEMG `detect_ventilator_breath`
- ReSurfEMG `find_occluded_breaths`
- ReSurfEMG `onoffpeak_baseline_crossing`
- ReSurfEMG `onoffpeak_slope_extrapolation`
- ReSurfEMG `find_linked_peaks`

Important constraint:

Do not force EIT and EMG into one breath detector. EIT still uses
valley-peak-valley impedance breaths. EMG still commonly starts with envelope
peaks and then derives start/end windows. The shared layer should provide the
pieces, not erase the domain logic.

Tests:

- Preserve EIT `BreathDetection.find_breaths` results on synthetic respiratory
  curves.
- Preserve ReSurfEMG detected peak indices on synthetic EMG/ventilator curves.
- Add adapter-level tests proving both outputs still coerce into `BreathEvent`.

### Phase 4: Shared Per-Breath Metrics

Add `m3resp.processing.metrics` with:

- `amplitude_at_peaks(values, peak_indices, baseline=None)`
- `time_to_peak(values, start_indices, end_indices, *, smooth=False)`
- `pseudo_slope(values, start_indices, end_indices, *, smooth=True)`
- `window_integral(values, sample_frequency, start_indices, end_indices, baseline=None)`
- `area_under_baseline(values, sample_frequency, peak_indices, start_indices, end_indices, window, baseline, reference_values=None)`
- `respiratory_rate_from_indices(indices, sample_frequency, *, outlier_percentile=33, outlier_factor=3)`
- `tidal_variation(values, time, breaths, *, method="inspiratory")`

Refactor targets:

- ReSurfEMG `postprocessing.features`
- EIT `TIV._calculate_tiv_values`
- Adapter conversions to `ParameterResult`

Tests:

- Compare ReSurfEMG feature outputs before/after.
- Compare continuous EIT TIV values before/after.
- Confirm `ParameterResult` metadata records source method and units.

### Phase 5: Quality Mapping

Add `m3resp.processing.quality` with small reusable helpers:

- `threshold_flag(name, value, *, threshold, comparison, modality=None)`
- `fraction_flag(name, passed_fraction, *, minimum_fraction, modality=None)`
- `timing_window_flag(name, deltas, *, min_delta=None, max_delta=None, modality=None)`

Refactor targets:

- `ReSurfEMGAdapter.to_quality_flags`
- ReSurfEMG quality-assessment wrappers
- Future EIT quality checks

Tests:

- Existing heterogeneous ReSurfEMG quality outputs still map into
  `QualityFlag`.
- Skipped postprocessing functions continue to become warning flags.

### Phase 6: Adapter Migration

After primitives are tested, update adapters incrementally:

1. Keep public adapter methods and keyword arguments unchanged.
2. Use shared primitives internally for new native `m3resp` paths.
3. Keep optional dependency behavior unchanged: installing bare `m3resp` must
   not import `eitprocessing` or `resurfemg` eagerly.
4. Keep custom callables (`preprocess=`, `detector=`, `compute=`,
   `postprocess=`) working.
5. Keep regression tests in `tests/regression/` as the guardrail for upstream
   equivalence.

## Compatibility Rules

- Do not change public `M3Session` method names or return shapes in the same
  commit as an extraction.
- Do not remove direct adapter calls to upstream packages until an equivalent
  native primitive has regression coverage.
- Do not move vendor file readers into generic processing modules.
- Do not move EIT ROI/pixel-map algorithms into generic EMG/EIT primitives.
- Do not make synchronization depend on post-detection event alignment. Raw
  synchronization remains the active path for multimodal alignment.

## Suggested Milestones

| Milestone | Deliverable | Verification |
|---|---|---|
| M1 | `m3resp.processing.filters` | Unit tests against EIT and EMG filter outputs |
| M2 | `m3resp.processing.windows` | Unit tests against moving average and RMS/ARV outputs |
| M3 | `m3resp.processing.peaks`/`intervals` | EIT breath and EMG peak regression tests |
| M4 | `m3resp.processing.metrics` | ReSurfEMG feature and EIT TIV regression tests |
| M5 | Quality helpers | `QualityFlag` conversion tests |
| M6 | Adapter cleanup | Existing `tests/` and `tests/regression/` pass |
| M7 | Documentation update | `migration.md`, `stage2.md`, and examples mention native primitives |

## First Pull Request

The first PR should be deliberately small:

1. Add `src/m3resp/processing/__init__.py`.
2. Add `src/m3resp/processing/filters.py`.
3. Add tests for Butterworth and notch behavior.
4. Update only the adapter code path that already performs native Butterworth
   filtering in `EITProcessingAdapter.preprocess`.
5. Leave ReSurfEMG's upstream functions untouched, but add equivalence tests
   showing the shared primitive can reproduce them.

This gives the project an immediate reusable primitive without changing the
scientific behavior of either upstream package.
