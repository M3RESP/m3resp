# Migrating from `eitprocessing`/`resurfemg` calls to `m3resp`

`m3resp` does not reimplement `eitprocessing` or `resurfemg` algorithms - it
calls them (see [stage1.md](stage1.md)'s dependency direction and the
regression tests in `tests/regression/`, which pin down that the wrappers
reproduce the underlying calls exactly). Migrating existing code means
replacing manual calls into those packages with the equivalent `m3resp` call,
not rewriting the science.

## EIT

| Direct `eitprocessing` call | `m3resp` equivalent |
|---|---|
| `eitprocessing.datahandling.loading.load_eit_data(path, vendor=...)` | `m3resp.io.load_eit(path, vendor=...)` or `session.load_eit(path, vendor=...)` |
| Manually chaining `RateDetection`, `MDNFilter`/`ButterworthFilter`, `BreathDetection`, `TIV`, `EELI` | `session.preprocess_eit(...)` (one call; see `EITProcessingAdapter.preprocess` for the exact parameters each stage maps to) |
| Reading `BreathDetection` output directly | `session.detect_eit_breaths()` -> `BreathEvent` objects |
| Reading `TIV`/`EELI`/rate results as raw `eitprocessing` objects | `session.preprocess_eit()` followed by `EITProcessingAdapter.to_parameters(processed)` -> `ParameterResult` objects, or just read `session.parameter_results` after `preprocess_eit()` |

## EMG

| Direct `resurfemg` call | `m3resp` equivalent |
|---|---|
| `resurfemg.data_connector.converter_functions.load_file(path)` | `m3resp.io.load_emg(path)` or `session.load_emg(path)` |
| `resurfemg.preprocessing.filtering.emg_bandpass_butter` + `resurfemg.preprocessing.envelope.full_rolling_arv` | `session.preprocess_emg(...)` |
| `resurfemg.postprocessing.event_detection.detect_emg_breaths` | `session.detect_emg_breaths()` -> `BreathEvent` objects |
| `resurfemg.postprocessing.features.*` / `resurfemg.postprocessing.quality_assessment.*` called by hand | `session.postprocess_emg(...)`, then read `session.parameter_results` (`ParameterResult`) and `session.quality` (`QualityFlag`) - populated via `ReSurfEMGAdapter.to_parameters`/`to_quality_flags` |
| Calling an arbitrary `resurfemg.postprocessing` function not covered above | `ReSurfEMGAdapter.run_postprocessing_function(category, function_name, *args, **kwargs)` |

## Cross-modality bookkeeping you previously did by hand

| Manual approach | `m3resp` equivalent |
|---|---|
| Tracking which package/version/parameters produced a file yourself | `session.provenance` (lightweight log) or attach `session.datamodel = DataModelRecorder(session)` for a full, validated `ProcessingRun`/`DataFile`/`DerivedFeature` audit trail (see [stage2.md](stage2.md)) |
| Shifting one recording's timestamps to match another by hand | `m3resp.compute_offsets_from_timestamps(reference_modality, timestamps)` then `m3resp.align_events_by_modality_offset` (or just `session.align_modalities(...)`) |
| Resampling one signal onto another's sampling rate by hand | `m3resp.resample_signal(signal, target_frequency_hz)` |
| Matching already-detected breaths (EIT/EMG/ventilator, or any other modality) by eyeballing timestamps | `m3resp.link_breaths_by_time({"eit": ..., "emg": ..., "ventilator": ...})` or `session.link_breaths(time_tolerance=...)` -> `LinkedBreath` objects (breath detection must already have produced the `BreathEvent`s passed in; this only matches breaths across modalities, it does not detect them) |
| Writing your own CSV/JSON export per project | `session.export_summary(output_dir)` (see [stage2.md](stage2.md#structured-export)) |

## What does not change

- `eitprocessing` and `resurfemg` remain independent, installable, and usable
  on their own - `m3resp` depends on them, never the reverse.
- Passing a custom callable (`preprocess=`, `detector=`, `compute=`) to the
  relevant `M3Session`/adapter method still works, for anything not covered
  by the built-in conversions.
