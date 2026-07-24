# Migrating from `eitprocessing`/`resurfemg` calls to `m3resp`

`m3resp` does not reimplement `eitprocessing` or `resurfemg` algorithms - it calls them (see [developer/architecture.md](developer/architecture.md)'s dependency direction and the regression tests in `tests/regression/`, which pin down that the wrappers reproduce the underlying calls exactly). Migrating existing code means replacing manual calls into those packages with the equivalent `m3resp` call, not rewriting the science.

Modality-specific migration tables:

- [migration/from-eitprocessing.md](migration/from-eitprocessing.md)
- [migration/from-resurfemg.md](migration/from-resurfemg.md)

## Cross-modality bookkeeping
| Manual approach | `m3resp` equivalent |
|---|---|
| Tracking which package/version/parameters produced a file yourself | `session.provenance` (lightweight log) or attach `session.datamodel = DataModelRecorder(session)` for a full, validated `ProcessingRun`/`DataFile`/`DerivedFeature` audit trail (see [concepts/provenance.md](concepts/provenance.md)) |
| Shifting one recording's timestamps to match another by hand | `m3resp.compute_offsets_from_timestamps(reference_modality, timestamps)` then `m3resp.align_events_by_modality_offset` (or just `session.align_modalities(...)`) |
| Resampling one signal onto another's sampling rate by hand | `m3resp.resample_signal(signal, target_frequency_hz)` |
| Matching already-detected breaths (EIT/EMG/ventilator, or any other modality) by eyeballing timestamps | `m3resp.link_breaths_by_time({"eit": ..., "emg": ..., "ventilator": ...})` or `session.link_breaths(time_tolerance=...)` -> `LinkedBreath` objects (breath detection must already have produced the `BreathEvent`s passed in; this only matches breaths across modalities, it does not detect them) |
| Computing a timing offset/delay between two modalities' breaths by hand | `session.compute_multimodal_parameters()` after `session.link_breaths()` - see [concepts/synchronization.md](concepts/synchronization.md) |
| Writing your own CSV/JSON export per project | `session.export_summary(output_dir)` (see [tutorials/export-results.md](tutorials/export-results.md)) |

## What does not change

- `eitprocessing` and `resurfemg` remain independent, installable, and usable on their own - `m3resp` depends on them, never the reverse.
- Passing a custom callable (`preprocess=`, `detector=`, `compute=`) to the relevant `M3Session`/adapter method still works, for anything not covered by the built-in conversions.
