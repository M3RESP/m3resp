# M3Resp Stage 2

Stage 2 turns `m3resp` from a thin wrapper around `eitprocessing`/`resurfemg`
(Stage 1, see [stage1.md](stage1.md)) into a shared multimodal data model and
framework, without breaking anything Stage 1 already provided. Every piece
below is additive: existing `M3Session` methods, the declarative pipeline
engine, and the CLI all still work exactly as documented in
[stage1.md](stage1.md) and [pipelines.md](pipelines.md).

Two design documents record the reasoning behind Stage 2 in detail:

- [`plan/plan_stage2.md`](../plan/plan_stage2.md) — the original Stage 2
  vision (milestones 2.1-2.8).
- [`plan/stage2_consolidation.md`](../plan/stage2_consolidation.md) — how that
  vision reconciles with the parallel data-model design in
  [`main_v0.3.tex`](../main_v0.3.tex)/`plan/data_model_stage2.md`, and with
  Stage 1's existing declarative pipeline engine. Read this first if
  something below looks like it duplicates existing functionality — it
  almost always doesn't, and the reasoning is written down there.

## The two data-model layers

Stage 2 has two layers of objects, and they are not competing designs:

```text
legacy package output (eitprocessing / resurfemg)
        |
        v
   adapter (EITProcessingAdapter / ReSurfEMGAdapter)
        |
        v
   Layer 1 - runtime objects (m3resp.data)
   Signal, ParameterResult, QualityFlag, LinkedBreath, Event/Breath
        |
        v
   DataModelRecorder (opt-in, session.datamodel)
        |
        v
   Layer 2 - persisted entities (m3resp.datamodel)
   SignalStream, DerivedFeature, QualityAnnotation, ProcessingRun, ...
        |
        v
   DataModelStore -> validate_store() / export_store()
```

- **Layer 1** (`m3resp.data`) is what processing code actually creates and
  passes around while a session runs.
- **Layer 2** (`m3resp.datamodel`) is a validated, queryable record of what
  happened - built for later consumption by things like the audit trail,
  export, or a future backend/GUI service layer.

## Package map: where to add new functionality

```text
src/m3resp/
├── core/            Session, events, exceptions, provenance, metadata
│   └── session.py     M3Session - see "Extending M3Session" below
│
├── data/            Layer 1: runtime scientific objects (Milestone 2.1/2.2/2.5)
│   ├── signals.py      Signal, TimeSeries - add new signal-shaped concepts here
│   ├── parameters.py   ParameterResult - add new computed-metric concepts here
│   ├── quality.py      QualityFlag
│   ├── linked_breath.py  LinkedBreath (cross-modality breath matching)
│   ├── processing.py   ProcessingStep / ProcessingHistory
│   └── collections.py  SignalCollection / ParameterResultCollection / QualityReport
│
├── datamodel/       Layer 2: persisted/audit entities (main_v0.3.tex's model)
│   ├── entities.py     Case, RecordingSession, SignalStream, DataFile, ...
│   ├── store.py        DataModelStore (in-memory, FK-checked tables)
│   ├── recorder.py     DataModelRecorder - the Layer 1 -> Layer 2 boundary
│   ├── validation.py    validate_store() - reference + completeness checks (doc Sec 10)
│   └── export.py       export_store() - one JSON file per table
│
├── adapters/        Conversion boundary to the legacy packages
│   ├── eitprocessing_adapter.py  load/preprocess + to_signals/to_parameters/
│   │                             to_quality_flags (Milestone 2.3)
│   └── resurfemg_adapter.py      same shape, for resurfemg
│
├── synchronization/ Alignment, resampling, breath linking (Milestone 2.5)
│   ├── alignment.py    manual-offset + timestamp-derived offsets
│   ├── resampling.py   resample_signal - common time base
│   └── linking.py      link_breaths_by_time - nearest-neighbor breath linking
│
├── workflows/       Stage 1's declarative step-registry engine (YAML/JSON specs)
│   └── steps/          add a new @register_step here for a custom, composable step
│
├── presets/         Named, built-in Pipeline presets (Milestone 2.4) - NOT the
│   │                same thing as workflows/ above; see presets/base.py
│   ├── eit.py, emg.py, multimodal.py   add a new preset here
│   └── registry.py     register_pipeline(name, cls)
│
├── modalities/      Top-level load helpers (load_eit, load_emg)
├── export/          session_export.py (Stage 1 + Milestone 2.6 structured export),
│                    tables.py (row-shaping helpers)
├── visualization/   Session overview and synchronization plots
└── synthetic/       Synthetic data generators for tests/examples
```

Rule of thumb for "where does my new EIT/EMG/multimodal functionality go":

1. **A new upstream algorithm you want exposed** -> a method on the adapter
   (`adapters/*.py`), converting its result to a Layer 1 object via
   `to_signals`/`to_parameters`/`to_quality_flags`.
2. **A new computed metric type** (not just a new instance of an existing
   one) -> `data/parameters.py` (`ParameterResult` already covers most cases;
   only add a new class if the concept genuinely isn't a named/valued/
   unit-tagged metric).
3. **A new composable pipeline step** for the YAML/JSON declarative engine
   -> `workflows/steps/*.py` with `@register_step`.
4. **A new one-call preset** ("run all of EIT/EMG/multimodal processing in
   one call") -> `presets/*.py`, registered in `presets/registry.py`.
5. **A new low-level, reusable synchronization building block** (e.g. a
   resampling method, an offset/alignment computation, a breath-linking
   strategy - something other code composes, not a full pipeline step or
   preset itself) -> `synchronization/`.
6. **A new persisted/audit entity** (something that needs to be queryable,
   validated, and exported later, per the `main_v0.3.tex` data model) ->
   `datamodel/entities.py`, wired into `datamodel/recorder.py`.

## Extending `M3Session`

`M3Session` (`core/session.py`) holds both the Stage 1 dict-based state
(`raw`, `processed`, `events`, `parameters` - unchanged) and the Stage 2 typed
collections added on top of it:

| Attribute | Populated by | Contains |
|---|---|---|
| `session.signals` | `preprocess_eit`/`preprocess_emg` (default adapter path) | `Signal` |
| `session.parameter_results` | `preprocess_eit`/`postprocess_emg` | `ParameterResult` |
| `session.quality` | `preprocess_eit`/`postprocess_emg` | `QualityFlag` |
| `session.linked_breaths` | `session.link_breaths()` | `LinkedBreath` |
| `session.datamodel` | opt-in: `session.datamodel = DataModelRecorder(session)` | Layer 2 entities |

A custom `preprocess=callable` passed to `preprocess_eit` bypasses the typed
collections (its output shape isn't guaranteed to match what the conversion
methods expect) - everything else about the session keeps working.

## Named pipelines vs. the declarative engine

Two different mechanisms both happen to be called "run a pipeline":

- `m3resp.run_pipeline(spec, session=...)` (module-level) runs a fully custom
  YAML/JSON step-list spec - the Stage 1 engine, documented in
  [pipelines.md](pipelines.md). Use this for bespoke or batch workflows.
- `session.run_pipeline("eit" | "emg" | "multimodal", config=...)` (a method
  on `M3Session`) runs one of the small, built-in `Pipeline` presets in
  `m3resp.presets`, each just a fixed sequence of calls to the session's
  own methods. Use this for the common case for one modality end-to-end.

## Structured export

`session.export_summary(output_dir)` writes the Stage 1 outputs (`summary.json`,
per-event-list CSVs, `parameters.csv`) plus, by default, the Milestone 2.6
structured files: `session_metadata.json`, `signals_manifest.csv`,
`parameter_results.csv`, `quality_flags.csv`, `linked_breaths.csv`, and
`processing_history.json`. Pass `structured_export=False` to skip the latter
group (matches the existing `event_csvs`/`parameters_csv`/`postprocessing`
toggles). Empty collections are skipped, not written as empty files.

For the persisted (Layer 2) data model, use `m3resp.export_store(store, dir)`
instead - one JSON file per entity table (`cases.json`, `sessions.json`, ...).

`m3resp.validate_store(store)` checks a Layer 2 store. By default it runs the
*reference* checks only: that every record links to records that exist and no
time window ends before it starts. A store recorded while a session runs passes
these. Pass `validate_store(store, require_complete=True)` before saving a
finished dataset to also check that each record carries the descriptive fields
the full data model expects (units, sampling rate, start time, file checksums).
A recorded store usually does *not* pass the completeness check yet, because a
live recording knows its time since the recording started but not the wall-clock
time it began, and a file is not checksummed until it is written to disk.

## Regression tests

`tests/regression/` pins down that the adapters are still thin wrappers:
each test drives the adapter's public API on synthetic data and asserts the
result is identical to calling the underlying `eitprocessing`/`resurfemg`
function directly with the same arguments. If one of these starts failing,
the adapter has started transforming data instead of just passing it
through - check the diff against the adapter method involved before assuming
the test is wrong.

## See also

- [stage1.md](stage1.md) - the Stage 1 wrapper layer these all build on.
- [pipelines.md](pipelines.md) - the declarative YAML/JSON pipeline spec format.
- [migration.md](migration.md) - calling `eitprocessing`/`resurfemg` directly vs. through `m3resp`.
