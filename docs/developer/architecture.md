# Architecture

Stage 2 turns `m3resp` from a thin wrapper around `eitprocessing`/`resurfemg`
([stage1.md](../stage1.md)) into a shared multimodal data model and
framework, without breaking anything Stage 1 already provided. Every piece
below is additive: existing `M3Session` methods, the declarative pipeline
engine, and the CLI all still work exactly as documented in
[stage1.md](../stage1.md) and [pipelines.md](../pipelines.md).

Stage 2 was designed in two parts that are not competing designs but two
layers of the same pipeline: the original Stage 2 vision (milestones
2.1-2.8), which this `docs/concepts/`, `docs/tutorials/`, `docs/migration/`,
`docs/developer/` layout follows; and how that vision reconciles with the
parallel persisted data model from `main_v0.3.tex` (`m3resp.datamodel`) and
with Stage 1's existing declarative pipeline engine. If something below
looks like it duplicates existing functionality, it almost always doesn't -
see "The two data-model layers" below for why the runtime (Layer 1) and
persisted (Layer 2) objects need to coexist.

## Plain-language overview

**Layer 1: runtime objects (`m3resp.data`)**

These are the objects described in [concepts/](../concepts/): `Signal`,
`ParameterResult`, `QualityFlag`, `LinkedBreath`, `Event`/`BreathEvent`.
"Runtime" means they exist only while code is actually running, in memory,
created fresh each time a session runs. They are lightweight (a dataclass
with a handful of fields, no database behind them) and their only job is to
be the common currency that flows between processing steps while a
pipeline executes.

**Layer 2: persisted entities (`m3resp.datamodel`)**

These are a different, heavier set of types: `Case`, `RecordingSession`,
`SignalStream`, `DataFile`, `ProcessingRun`, `DerivedFeature`,
`QualityAnnotation`, and more. "Persisted" means they are meant to be saved
and looked up later, not just used transiently mid-computation. They live
inside a `DataModelStore` (an in-memory table structure that behaves like a
small, simplified database: each entity type is a table, and it checks
foreign keys, meaning it verifies that if one record refers to another
record by ID, that other record actually exists). This layer supports two
extra operations the runtime objects do not have: `validate_store()`
(checks the data is complete and internally consistent) and
`export_store()` (writes everything out, one JSON file per table).

### How you move from Layer 1 to Layer 2

Layer 2 is opt-in. By default, a session only ever produces Layer 1
objects. If you want the persisted layer as well, you attach a
`DataModelRecorder` to the session
(`session.datamodel = DataModelRecorder(session)`). From that point on,
this recorder acts as the boundary or translator: it watches what Layer 1
produces and converts it into the matching Layer 2 entity. This is why the
doc below stresses "not competing designs, two layers of the same
pipeline": it is one straight-line flow of data, just with an optional
second stop at the end, not two separate systems fighting for the same
job.

### Why two layers instead of one

The reason comes down to these two points: we are making fundamentally different trade-offs.

- Layer 1 needs to be fast and cheap: every processing step creates and
  touches these objects constantly, so they should not carry validation
  overhead, foreign-key checks, or a rigid schema that could slow things
  down or force every intermediate step to be "complete" before it is even
  fully computed.
- Layer 2 needs to be strict and complete: an audit trail is only useful
  if you can trust it, so it enforces things like every reference pointing
  to a real record, and (via `validate_store(require_complete=True)`) that
  important descriptive fields like units, sampling rate, and file
  checksums are actually filled in before the dataset is considered
  finished.

Trying to merge them into one type would force a bad compromise either
way: make every in-flight processing object carry full validation and
relational bookkeeping (slow and premature, since a signal mid-pipeline is
not "complete" yet), or strip the audit layer down to something as loose
as the runtime objects (which would defeat the point of an audit trail).
So the design keeps them as two separate types connected by one converter
(`DataModelRecorder`), a common pattern known as separating the "domain
model" (what the code works with while running) from the "persistence
model" (what gets saved and validated for later use).

### Why Layer 2 exists at all, concretely

Layer 2 is built for consumers that are not the processing code itself:

- Reproducibility/audit: proving exactly what recording, what processing
  run, and what derived features produced a given result, in a format that
  can be checked for consistency.
- Export: `export_store()` turns the whole thing into a stable, portable
  set of JSON files.
- Future GUI/service layer: since these entities are already validated and
  relational, a future application can query them directly instead of
  re-deriving that structure from raw Layer 1 objects each time.

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

- **Layer 1** (`m3resp.data`, described in [concepts/](../concepts/)) is what
  processing code actually creates and passes around while a session runs.
- **Layer 2** (`m3resp.datamodel`) is a validated, queryable record of what
  happened - built for later consumption by things like the audit trail,
  export, or a future backend/GUI service layer.

## Package map: where to add new functionality

```text
src/m3resp/
├── core/            Session, events, exceptions, provenance, metadata
│   └── session.py     M3Session - see concepts/session.md
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
├── adapters/        Conversion boundary to the legacy packages - see adapters.md
│   ├── eitprocessing_adapter.py  load/preprocess + to_signals/to_parameters/
│   │                             to_quality_flags (Milestone 2.3)
│   └── resurfemg_adapter/        same shape, for resurfemg (split by
│                                  responsibility: core/ecg/baseline/quality/defaults)
│
├── synchronization/ Alignment, resampling, breath linking, multimodal
│   │                parameters (Milestone 2.5, see concepts/synchronization.md)
│   ├── alignment.py    manual-offset + timestamp-derived offsets
│   ├── offset_estimation.py  cross-correlation-based offset estimation
│   ├── timebase.py     Timebase - common time-axis representation
│   ├── resampling.py   resample_signal - common time base
│   ├── linking.py      link_breaths_by_time - nearest-neighbor breath linking
│   ├── cropping.py     raw-modality offset resolution + in-place cropping,
│   │                   used by M3Session.synchronize_raw_modalities
│   ├── ventilator.py   ventilator breath-detection normalization into BreathEvents
│   └── multimodal_parameters.py  compute_timing_delay / compute_event_agreement /
│                        compute_breath_duration_difference / compute_multimodal_parameters
│
├── workflows/       Stage 1's declarative step-registry engine (YAML/JSON specs)
│   └── steps/          add a new @register_step here for a custom, composable step
│       ├── eit/         eit.* steps, split by pipeline stage (filtering/pixel/roi/loading/signals)
│       └── emg/         emg.* steps, split by pipeline stage (baseline/ecg_*/features/quality_*/...)
│
├── presets/         Named, built-in Pipeline presets (Milestone 2.4) - see
│   │                developer/pipeline-contracts.md; NOT the same thing as
│   │                workflows/ above; see presets/base.py
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
5. **A new low-level, reusable synchronization or multimodal-metric building
   block** (e.g. a resampling method, an offset/alignment computation, a
   breath-linking strategy, a cross-modality timing metric - something other
   code composes, not a full pipeline step or preset itself) ->
   `synchronization/`.
6. **A new persisted/audit entity** (something that needs to be queryable,
   validated, and exported later, per the `main_v0.3.tex` data model) ->
   `datamodel/entities.py`, wired into `datamodel/recorder.py`.
7. **A completely new algorithm with no upstream equivalent** (not wrapping
   `eitprocessing`/`resurfemg`, genuinely new science) -> out of scope for
   Stage 2, which only wraps existing upstream behavior; see "Stage 3
   outlook" below for where this goes once Stage 3's native packages exist.

## Stage 3 outlook: what evolves, what stays, what goes

This section maps the pieces above onto Stage 3, based on the Stage 3
sections in `plan/stage2/0_remaining_gap_migration_plan.md`,
`1_eit_gap_migration_implementation_plan.md`,
`2_resurfemg_gap_migration_implementation_plan.md`, and
`3_pipeline_structure_implementation_plan.md`.

### Ready to use as-is (stable contracts, no change needed)

These were deliberately built in Stage 2 to be backend-neutral, so Stage 3
does not need to touch them:

- **Layer 1 runtime objects** (`data/signals.py`, `parameters.py`,
  `quality.py`, `linked_breath.py`, `processing.py`): `Signal`,
  `ParameterResult`, `QualityFlag`, `LinkedBreath`, `Event`/`BreathEvent`.
  Their whole design point was to be a shared, upstream-independent shape
  both modalities produce, so they carry over unchanged.
- **`workflows/`** (the declarative step-registry engine): the plan states
  explicitly to keep `m3resp.workflows` as the canonical Stage 2 and Stage 3
  public module. The YAML/JSON spec format, the registry, and the engine
  stay exactly as they are.
- **`datamodel/`** (Layer 2, persisted entities): `Case`,
  `RecordingSession`, `ProcessingRun`, `DataModelStore`, `validate_store()`,
  `export_store()`. This layer only cares about recording what happened,
  never about which backend did the computing, so it is unaffected by the
  upstream swap.
- **`synchronization/`**: alignment, resampling, breath linking,
  multimodal parameters. These operate purely on `Signal`/`BreathEvent`
  objects, not on upstream library objects, so they are already
  backend-neutral.
- **`M3Session`'s public method names and signatures**: `load_eit`,
  `preprocess_eit`, `detect_eit_breaths`, `link_breaths`, `export_summary`,
  and the rest. This is called out as a stable contract specifically so
  Stage 3 does not have to touch calling code.
- **`presets/`**: these just call the session's own already-instrumented
  methods in sequence, so they ride along with whatever `M3Session` does
  internally.
- **Provenance schema**: `ProvenanceRecord`, `ProcessingStep`/
  `ProcessingHistory`, and the `metadata.operation` field in provenance
  records. The plan is explicit that `metadata.operation` (for example
  `"eit.pixel_tiv"`) is the stable identifier for workflows and the GUI
  across Stage 2 and Stage 3.

### Replaced under the hood (same public shape, different internals)

These keep their name, their method signature, and their place in the
package map, but what runs inside them changes:

- **`adapters/eitprocessing_adapter.py` and `adapters/resurfemg_adapter/`**:
  today these wrap calls into the `eitprocessing`/`resurfemg` libraries.
  Stage 3 replaces what is inside them, one operation at a time, with calls
  into new native packages: `src/m3resp/eit/io/`, `eit/processing/`,
  `eit/roi/`, `emg/io/`, `emg/processing/`. The plan calls the current
  adapters a temporary Stage 2 backend, not the public GUI contract, while
  `M3Session` and the workflow steps are the real stable contract.
- **`modalities/`** (top-level `load_eit`/`load_emg` helpers): these
  currently call `adapter.load()`, which calls a vendor-specific upstream
  reader. In Stage 3 they call the new native readers in `eit/io/`/
  `emg/io/` instead, in this dependency order per the plan: vendor
  loading/normalization first, then native containers and global
  impedance, then breath/rate detection, then filtering (MDN), then
  EELI/TIV, then pixel-level and ROI behavior.
- **`workflows/steps/eit/*.py` and `steps/emg/*.py`**: today these bind
  context keys and parameters, then call `session.eit_adapter`/
  `session.emg_adapter`. In Stage 3 they call the native services directly
  instead. The step names, `reads`/`writes` bindings, and registry metadata
  stay the same, only the function they delegate to changes.
- **Provenance metadata content**: fields like `method`,
  `metadata.source_package`, and `metadata.source_function` currently name
  `eitprocessing`/`resurfemg` classes and functions. In Stage 3 these get
  renamed to name the native `m3resp` implementation instead. The old
  upstream info is not discarded, it is kept under a renamed `reference_*`
  field so scientific equivalence stays traceable, per the plan.
- **Layer 1 dict slots that still hold upstream objects**
  (`session.processed["eit"]`, `processed_variants`): during Stage 2 these
  hold the original upstream object side by side with the native
  `Signal`/`ParameterResult` conversion, for backward compatibility and
  regression comparison. In Stage 3, once an operation is fully native, the
  upstream object is dropped from that slot; only the native `m3resp` type
  remains under the same context key.

### New algorithms with no upstream equivalent

This is a different case from everything above: it is not a Stage 2 piece
that Stage 3 changes, it is something Stage 2 cannot support at all. A
completely new algorithm (not a wrapper around existing `eitprocessing`/
`resurfemg` behavior) has nowhere to go in Stage 2, since Stage 2's adapters
only exist to wrap upstream calls (item 7 in the "Rule of thumb" list
above). Once Stage 3's native `eit/processing/`/`emg/processing/` packages
exist, a new algorithm is added directly there as ordinary native code, with
no adapter step, since there is no upstream call left to wrap. It only
belongs in the shared `m3resp.processing` package instead of a
modality-owned one if it is genuinely modality-neutral, by the same test
already applied to the existing shared primitives (`filters`, `peaks`,
`windows`, `intervals`, `metrics`).

### Removed fully

- **The runtime/production dependency on `eitprocessing` and
  `resurfemg`**: the Stage 3 completion gate states that no production
  source import or dependency may reference `eitprocessing` or
  `resurfemg`. A plain `pip install m3resp` and the distributed GUI will
  not install or import either package at all.
- **The reference packages as installable extras for normal users**: they
  move from being an optional runtime extra (as in Stage 2, where they are
  needed to actually run the delegated algorithms) to a
  development/reference-test-only extra, used solely to run an optional
  comparison suite against the frozen Stage 2 golden fixtures.
- **Any direct exposure of upstream objects to calling code or a GUI**: the
  plan is explicit that the future GUI must not import from
  `m3resp.adapters`, `eitprocessing`, or `resurfemg`, and must never
  receive an upstream `Sequence`, `EITData`, `SparseData`, `PixelMask`, or
  ReSurfEMG object. That entire code path (upstream object flowing out to a
  caller) is eliminated, not just deprioritized.

The adapter classes themselves
(`EITProcessingAdapter`/`ReSurfEMGAdapter`) are not necessarily deleted
outright, since adapter injection remains available for regression tests in
Stage 3, meaning they likely stick around as a reference-comparison harness
even after production code stops calling them.

## See also

- [concepts/](../concepts/) - what each Layer 1 object is and what populates it.
- [tutorials/](../tutorials/) - end-to-end walkthroughs using these objects.
- [adapters.md](adapters.md) - the adapter conversion boundary in detail.
- [pipeline-contracts.md](pipeline-contracts.md) - `Pipeline`/presets vs. the declarative engine.
- [testing.md](testing.md) - regression tests and the test layout.
- [../pipelines.md](../pipelines.md) - the declarative YAML/JSON pipeline spec format.
- [../migration/](../migration/) - calling `eitprocessing`/`resurfemg` directly vs. through `m3resp`.
