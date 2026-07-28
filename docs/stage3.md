# M3Resp Stage 3

Stage 3 is not implemented yet. This page is a rough picture of its destination and boundaries. Treat it as a planning document (a description of intent, not working code yet), not a finished spec (a precise, binding technical description): exact module boundaries (where one folder/package's responsibility ends and another's begins) "may be adjusted during the Stage 3 architecture work" per the plan itself.

Stage 1 ([stage1.md](stage1.md)) made `m3resp` a thin wrapper (a layer of code that mostly just forwards calls to another library, adding little of its own) around `eitprocessing`/`resurfemg`. Stage 2 ([stage2.md](stage2.md)) unified the public surface (the set of names/functions other code is meant to call, as opposed to internal details that can change freely): `M3Session`, native `Signal`/`ParameterResult`/`QualityFlag`/`BreathEvent` types (native meaning implemented directly in this codebase, not borrowed from an outside library), and the declarative workflow engine (a system where you describe *what* steps to run in a YAML/config file, rather than writing the step-by-step Python code yourself). Stage 2 still delegates the actual scientific computation to those two upstream (external, maintained-elsewhere) packages behind adapters (wrapper objects that translate calls from `m3resp` into calls on an outside library). Stage 3 removes that delegation: the EIT and EMG algorithms move into `m3resp` itself, and the project stops depending on `eitprocessing`/`resurfemg` at runtime.

## Goal

Make `m3resp` standalone (able to run fully on its own, without needing another *specialist* package installed): EIT and EMG readers, processing, metrics, quality behavior, and GUI-facing services (the backend functions a graphical user interface calls) all live in `m3resp`, with no production runtime dependency (something the finished, shipped software needs installed to run) on `eitprocessing` or `resurfemg`. The general scientific stack (`numpy`/`scipy`) stays a normal dependency either way; only the two dedicated EIT/EMG packages go away. A GUI can then be built on top of `m3resp` without ever importing either upstream package.

## What "merging EIT and EMG" actually means

The whole point of `m3resp` is to merge EIT and EMG analysis into one integrated tool: one `M3Session` object holds both modalities' data for the same recording, one pipeline engine runs both modalities' steps from the same YAML spec, and one set of shared data types (`Signal`, `ParameterResult`, `QualityFlag`, `BreathEvent`) lets EIT and EMG results be compared, time-aligned, and linked breath-by-breath. That integration is real and is the reason the project exists.

What Stage 3 does *not* do is fuse the two algorithm libraries' source code into one shared implementation. `eitprocessing`'s EIT algorithms and `resurfemg`'s EMG algorithms solve different scientific problems (impedance imaging vs. muscle electrical signals), so once ported natively they still land in separate, modality-owned folders (`src/m3resp/eit/` vs `src/m3resp/emg/`), not merged into one indistinguishable algorithm library. The "merge" is at the framework level (one session, one pipeline engine, one set of data types), not at the level of the two libraries' internal math becoming the same code.

The two libraries do already share a small handful of genuinely modality-neutral building blocks, already factored out into `m3resp.processing` in Stage 2 and reused by both sides:

- `m3resp.processing.filters` - generic Butterworth filtering (lowpass/highpass/bandpass/bandstop), used by both EIT and EMG signal cleanup.
- `m3resp.processing.peaks` - generic peak detection in an array, used for both EIT breath detection and EMG peak/ECG detection.
- `m3resp.processing.windows` - rolling-window and envelope calculations, used by EMG baseline steps and reusable anywhere a moving average/window is needed.
- `m3resp.processing.intervals` - baseline-crossing and onset/offset detection, used to turn a continuous signal into discrete start/end events for a breath or other interval, regardless of modality.
- `m3resp.processing.metrics` - generic per-breath metrics (e.g. integrating a signal over a time interval), once a breath interval already exists, regardless of which modality detected it.

These are the only things that get shared as one common implementation; everything else, vendor readers, breath detection specifics, filtering choices, ROI (region-of-interest) definitions, clinical quality thresholds, stays owned by `eit/` or `emg/` and is not merged between the two.

## What moves, and where

Stage 2 already deliberately keeps modality-specific behavior (code that only makes sense for one kind of data, EIT or EMG, not both) out of generic helpers (shared utility code meant to serve many callers), so the target layout for native code is modality-owned (each folder belongs to and is only about one modality) rather than one big shared "processing" bucket:

```text
src/m3resp/eit/io/           EIT vendor readers (Draeger, Sentec, Timpel)
src/m3resp/eit/processing/   rate, breath, MDN, EELI, TIV, pixel behavior
src/m3resp/eit/roi/          EIT lung-space and ROI behavior
src/m3resp/emg/io/           EMG and ventilator readers
src/m3resp/emg/processing/   filtering, ECG removal, events, metrics, quality
src/m3resp/gui/              GUI built only on public m3resp services
```

Only the genuinely reusable, scientifically-neutral primitives listed above live outside these modality-owned packages. Anything that encodes EIT- or EMG-specific policy (a rule or default choice specific to one modality: vendor quirks, clinical thresholds, ROI definitions) stays under `eit/` or `emg/`, not in a generic (shared, modality-unaware) module (a single Python file/package).

Suggested native port order (the sequence to reimplement things in, since later pieces build on earlier ones), starting with EIT:

1. Vendor loading and normalization (Draeger, Sentec, Timpel; normalization here means putting different vendors' raw formats into one consistent shape).
2. Native EIT containers (the data-holding objects/classes) and global impedance.
3. Rate and global breath detection.
4. Filtering, including MDN.
5. Continuous EELI and TIV.
6. Pixel-breath and pixel-TIV calculations.
7. TIV, amplitude, watershed, and size-filtered ROI behavior.
8. Native plotting/view models needed by the GUI.

EMG follows the same approach in its own Stage 3 plan, sharing only the primitives listed above whose equivalence (producing the same result) with `resurfemg` has already been proven in Stage 2 regression tests (automated tests that re-check old behavior still holds after a change). In both cases, backend operations are replaced one at a time, keeping the public workflow contract (the promised names/inputs/outputs other code relies on) unchanged throughout.

## What stays as-is

Stage 2 built several things specifically so Stage 3 would not need to touch them. See [developer/architecture.md](developer/architecture.md#stage-3-outlook-what-evolves-what-stays-what-goes) for the full breakdown; in short:

- `M3Session`'s public method names and signatures (a signature is a function's name plus its expected inputs/outputs; keeping it the same means existing calling code keeps working unmodified).
- The Layer 1 runtime objects (`Signal`, `ParameterResult`, `QualityFlag`, `BreathEvent`, `LinkedBreath`), meaning the lightweight data objects created fresh each time code runs, as opposed to anything saved to disk.
- `m3resp.workflows`, the declarative step-registry engine (the system, described above, that runs named, YAML-described steps), which stays the canonical (the one official, intended-to-be-used) public module across both Stage 2 and Stage 3.
- The Layer 2 persisted/audit data model (`m3resp.datamodel`): a "persisted" record is one meant to be saved and looked up later; an "audit" record is one built so someone can later verify exactly what was done.
- The provenance schema (a record of what produced a result and how, with a fixed set of fields that record always has), in particular the stable `metadata.operation` identifier (a fixed label naming which operation was run, so tools can rely on it not changing) used by workflows and the GUI.

## Dependency transition

- `eitprocessing`/`resurfemg` stop being a production/runtime extra. A plain `pip install m3resp` and the distributed GUI must not install or import either package.
- The reference packages (the original external libraries, kept around only for comparison purposes) move to a development/reference-test-only extra, used solely to run an optional comparison suite (test group) against the frozen Stage 2 golden fixtures (saved, trusted example inputs/outputs used as the standard to compare new results against).
- Stage 2 characterization fixtures (test data that pins down exactly how the old, upstream-backed behavior worked, so it can be checked for a match later) must remain runnable in Stage 3 without installing the upstream packages at all.

## GUI boundary

The GUI is new in Stage 3 and must:

- Discover operations and controls from the `m3resp` step/capability registry (a lookup list of every available operation and what it needs, that code can query at runtime), not by importing scientific implementation classes directly.
- Call an `m3resp` application/service API, never an adapter or a scientific implementation class.
- Exchange only native data objects and serializable configuration (meaning it can be converted to a plain, portable format like JSON, with nothing exotic like a live database connection or a running process attached).
- Show units, parameter limits, warnings, provenance, and unavailable-input reasons, all supplied by `m3resp`.
- Support loading, preprocessing, event detection, metrics, quality review, visualization, and export for both EIT and EMG.
- Remain testable with synthetic native fixtures (artificially generated sample data), with no upstream package installed.

The GUI must never import from `m3resp.adapters`, `eitprocessing`, or `resurfemg`, and must never receive an upstream `Sequence`, `EITData`, `SparseData`, `PixelMask`, or ReSurfEMG object (all of these are data types belonging to the outside libraries, not to `m3resp`). GUI-specific state (data that only matters for how the screen currently looks, like which button is selected) must not leak back into scientific processing functions (i.e. the science code must stay usable and correct even with no GUI running at all).

## Completion gate

Stage 3 is done only when:

- The full EIT, EMG, and multimodal (multiple data types combined, here EIT + EMG together) workflows pass using native implementations.
- Characterization tests demonstrate parity (an exact or tolerance-bounded match) with the frozen Stage 2 reference behavior.
- No production source import or dependency (no line of shipped code that imports or requires) references `eitprocessing` or `resurfemg`.
- The GUI runs end to end (from first user action through to a finished result, with every step in between working) against native `m3resp` services only.

## Out of scope / not yet decided

- The exact internal folder layout under `eit/`/`emg/` may still change during Stage 3 architecture work; the ownership boundaries (which package is responsible for which behavior) are the firm part, not the file paths.
- Whether `EITProcessingAdapter`/`ReSurfEMGAdapter` are deleted outright or kept as a thin regression/reference-comparison harness (a minimal test setup built only to run comparisons, not part of the real product) is undecided; adapter injection (passing in a specific adapter object, typically for testing) is expected to remain available for regression tests even after production code stops calling it.
- Stage 3 does not fuse `eitprocessing`'s and `resurfemg`'s algorithms into one shared implementation, see ["What 'merging EIT and EMG' actually means"](#what-merging-eit-and-emg-actually-means) above; it ports each modality's validated behavior into its own `eit`/`emg` module, sharing only the small set of scientifically-neutral primitives listed there.
- Whether `m3resp.processing` primitives (`filters`, `windows`, `ecg`, ...) gain a `Signal`/`TimeSeries`-accepting convenience layer, so a caller can pass one object instead of an array plus `sample_frequency` separately, is undecided. Today these stay deliberately dependency-free (no import of `m3resp.data`), since most current call sites operate on raw arrays - sometimes upstream `eitprocessing`/`resurfemg` objects - before any `Signal`/`TimeSeries` exists; revisit this once Stage 3's native EIT/EMG containers make it clearer what's actually available at each call site. If this does get built, the shape discussed on [PR #24](https://github.com/M3RESP/m3resp/pull/24) is a thin adapter *on top of* the existing low-level functions (e.g. `apply_to_signal(lowpass_filter, signal)` or a `signal_handler_wrapper` decorator) living in `m3resp.data`, which already depends on `m3resp.processing` - not a change to the low-level functions themselves, which would reintroduce the upward dependency this entry rules out.

## See also

- [stage1.md](stage1.md) - the wrapper layer Stage 3 eventually removes the runtime dependency on.
- [stage2.md](stage2.md) - the contracts and native types Stage 3 builds on.
- [developer/architecture.md](developer/architecture.md) - package map and a detailed per-piece Stage 3 evolution table.
