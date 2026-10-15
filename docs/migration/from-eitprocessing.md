# Migrating from direct `eitprocessing` calls

`m3resp` does not reimplement `eitprocessing` algorithms - it calls them (see [../developer/architecture.md](../developer/architecture.md)'s dependency direction and the regression tests in `tests/regression/`, which pin down that the wrappers reproduce the underlying calls exactly). Migrating existing code means replacing manual calls into `eitprocessing` with the equivalent `m3resp` call, not rewriting the science.

| Direct `eitprocessing` call | `m3resp` equivalent |
|---|---|
| `eitprocessing.datahandling.loading.load_eit_data(path, vendor=...)` | `m3resp.io.load_eit(path, vendor=...)` or `session.load_eit(path, vendor=...)` |
| Manually chaining `RateDetection`, `MDNFilter`/`ButterworthFilter`, `BreathDetection`, `TIV`, `EELI` | `session.preprocess_eit(...)` (one call; see `EITProcessingAdapter.preprocess` in [../developer/adapters.md](../developer/adapters.md) for the exact parameters each stage maps to) |
| Reading `BreathDetection` output directly | `session.detect_eit_breaths()` -> `BreathEvent` objects (see [../concepts/events-and-breaths.md](../concepts/events-and-breaths.md)) |
| `IntervalData` (values per interval) and `SparseData` (values at single times) | `m3resp.IntervalData` and `m3resp.EventData`; `coerce_intervals(...)` turns eitprocessing's `Interval` objects into m3resp `Interval` objects, and `coerce_events(...)` does the same for single-time events (see [../concepts/events-and-breaths.md](../concepts/events-and-breaths.md#values-per-interval-or-event)) |
| Reading `TIV`/`EELI`/rate results as raw `eitprocessing` objects | After `session.preprocess_eit()`, read the rates from `session.parameter_results` (see [../concepts/parameters.md](../concepts/parameters.md)) and TIV, EELI and pixel TIV, one value per breath, from `session.interval_data` (see [../concepts/events-and-breaths.md](../concepts/events-and-breaths.md#values-per-interval-or-event)) |

See [../tutorials/eit-only.md](../tutorials/eit-only.md) for a full end-to-end example, and [../migration.md](../migration.md) for cross-modality bookkeeping that applies to EIT the same way it does to EMG.

## What does not change

`eitprocessing` remains independent, installable, and usable on its own - `m3resp` depends on it, never the reverse. Passing a custom `preprocess=callable`/`detector=callable` to the relevant `M3Session`/adapter method still works, for anything not covered by the built-in conversions.
