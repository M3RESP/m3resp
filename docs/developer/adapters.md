# Adapters

`m3resp` does not reimplement EIT/EMG algorithms - it calls
`eitprocessing`/`resurfemg` and converts their output into the Layer 1
objects described in [concepts/](../concepts/). The adapter is the
conversion boundary: everything on the near side of it (`M3Session`,
`m3resp.data`, exports) only ever sees `Signal`/`ParameterResult`/
`QualityFlag`/`BreathEvent`, never a raw `eitprocessing`/`resurfemg` object.

```text
eitprocessing output
    |
    v
EITProcessingAdapter
    |
    v
m3resp Signal / BreathEvent / ParameterResult / QualityFlag
```

```text
resurfemg output
    |
    v
ReSurfEMGAdapter
    |
    v
m3resp Signal / BreathEvent / ParameterResult / QualityFlag
```

## `EITProcessingAdapter` (`src/m3resp/adapters/eitprocessing_adapter.py`)

| Method | Responsibility |
|---|---|
| `load(path, vendor, **kwargs)` | Load a vendor EIT file into an `eitprocessing` sequence. |
| `get_raw_eit` / `get_global_impedance` | Extract pixel/global impedance from a loaded sequence. |
| `detect_rates`, `apply_mdn`, `find_pixel_breaths`, `compute_eeli`, `compute_pixel_tiv`, `compute_tiv_lungspace`, `compute_amplitude_lungspace`, `compute_watershed_lungspace`, `filter_roi_by_size` | Individual `eitprocessing` operations, exposed one-to-one so a caller can compose a custom sequence. |
| `preprocess(...)` | Chains the operations above into one call - what `session.preprocess_eit()` uses by default. |
| `detect_breaths(data, **kwargs)` | Returns `list[BreathEvent]` directly - already converted. |
| `compute_tiv(sequence, **kwargs)` | Runs TIV computation on a loaded sequence. |
| `to_signals(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[Signal]`. |
| `to_parameters(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[ParameterResult]` (TIV, EELI, rate, ...). |
| `to_quality_flags(preprocessed)` | **Conversion boundary.** Turns a `preprocess()` result into `list[QualityFlag]`. |

## `ReSurfEMGAdapter` (`src/m3resp/adapters/resurfemg_adapter.py`)

| Method | Responsibility |
|---|---|
| `load(path, **kwargs)` | Load an EMG/ventilator file. |
| `preprocess(signal, **kwargs)` | Filtering + envelope, matching `resurfemg.preprocessing`. |
| `detect_breaths(signal, **kwargs)` | Returns `list[BreathEvent]` directly - already converted. |
| `compute_features(...)` | Amplitude/AUC/pseudo-slope/time-to-peak and related per-breath features. |
| `to_signals(processed_emg)` | **Conversion boundary.** Turns a `preprocess()`/`postprocess()` result into `list[Signal]`. |
| `to_parameters(postprocessed)` | **Conversion boundary.** Turns a `postprocess()` result into `list[ParameterResult]`. |
| `to_quality_flags(postprocessed)` | **Conversion boundary.** Turns a `postprocess()` result into `list[QualityFlag]` (native `resurfemg` clinical quality checks). |
| `available_postprocessing()` / `postprocess(...)` / `run_postprocessing_function(category, function_name, ...)` | Discover and call `resurfemg.postprocessing` functions not covered by a named wrapper above, without leaving the adapter boundary. |
| `detect_ecg_peaks`, `gate_ecg`, `wavelet_denoise_ecg`, `moving_baseline`, `slopesum_baseline`, `snr_pseudo`, `pocc_quality`, `interpeak_distance`, `percentage_under_baseline`, `detect_local_high_aub`, `detect_extreme_time_products`, `detect_non_consecutive_manoeuvres`, `evaluate_bell_curve_error`, `evaluate_event_timing`, `evaluate_respiratory_rates` | Individual ECG-removal, baseline, and clinical quality operations, exposed one-to-one for the declarative pipeline engine (`workflows/steps/emg.py`) and custom composition. |

## Regression guarantee

`tests/regression/` pins down that the adapters are still thin wrappers:
each test drives the adapter's public API on synthetic data and asserts the
result is identical to calling the underlying `eitprocessing`/`resurfemg`
function directly with the same arguments. If one of these starts failing,
the adapter has started transforming data instead of just passing it
through - check the diff against the adapter method involved before assuming
the test is wrong. See [testing.md](testing.md).

## Adding a new upstream algorithm

1. Add a thin wrapper method on the relevant adapter that calls the upstream
   function with the same argument names/defaults it has.
2. If the result should be visible on `M3Session`'s typed collections, extend
   `to_signals`/`to_parameters`/`to_quality_flags` (or add a new `to_*`
   method if the shape doesn't fit those three).
3. Add a regression test asserting your wrapper's output matches calling the
   upstream function directly.
4. Optionally register a `workflows/steps/*.py` step so the operation is also
   reachable from a declarative YAML/JSON pipeline (see
   [pipeline-contracts.md](pipeline-contracts.md)).
