# Migrating from direct `resurfemg` calls

`m3resp` does not reimplement `resurfemg` algorithms - it calls them (see [../developer/architecture.md](../developer/architecture.md)'s dependency direction and the regression tests in `tests/regression/`, which pin down that the wrappers reproduce the underlying calls exactly). Migrating existing code means replacing manual calls into `resurfemg` with the equivalent `m3resp` call, not rewriting the science.

| Direct `resurfemg` call | `m3resp` equivalent |
|---|---|
| `resurfemg.data_connector.converter_functions.load_file(path)` | `m3resp.io.load_emg(path)` or `session.load_emg(path)` |
| `resurfemg.preprocessing.filtering.emg_bandpass_butter` + `resurfemg.preprocessing.envelope.full_rolling_arv` | `session.preprocess_emg(...)` - note the default band-pass is 20-500 Hz and the default envelope is **RMS**, not ARV; pass `envelope_method="arv"` to reproduce `full_rolling_arv` exactly |
| `resurfemg.postprocessing.event_detection.detect_emg_breaths` | `session.detect_emg_breaths()` -> `BreathEvent` objects (see [../concepts/events-and-breaths.md](../concepts/events-and-breaths.md)) |
| `resurfemg.postprocessing.features.*` / `resurfemg.postprocessing.quality_assessment.*` called by hand | `session.postprocess_emg(...)`, then read `session.parameter_results` (`ParameterResult`) and `session.quality` (`QualityFlag`) - populated via `ReSurfEMGAdapter.to_parameters`/`to_quality_flags` (see [../developer/adapters.md](../developer/adapters.md)) |
| Calling an arbitrary `resurfemg.postprocessing` function not covered above | `ReSurfEMGAdapter.run_postprocessing_function(category, function_name, *args, **kwargs)`, or `session.emg_adapter.run_postprocessing_function(...)` |
| Manual ECG gating/wavelet-denoising/estimated-subtraction calls | The corresponding `ReSurfEMGAdapter` method (`gate_ecg`, `wavelet_denoise_ecg`, ...), or the matching `emg.*` step in the declarative pipeline engine - see "ECG-removal alternatives" in [../pipelines.md](../pipelines.md) |

See [../tutorials/emg-only.md](../tutorials/emg-only.md) for a full end-to-end example, and [../migration.md](../migration.md) for cross-modality bookkeeping that applies to EMG the same way it does to EIT.

## What does not change

`resurfemg` remains independent, installable, and usable on its own - `m3resp` depends on it, never the reverse. Passing a custom `preprocess=callable`/`detector=callable`/`compute=callable` to the relevant `M3Session`/adapter method still works, for anything not covered by the built-in conversions.
