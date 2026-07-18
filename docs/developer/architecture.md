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

## See also

- [concepts/](../concepts/) - what each Layer 1 object is and what populates it.
- [tutorials/](../tutorials/) - end-to-end walkthroughs using these objects.
- [adapters.md](adapters.md) - the adapter conversion boundary in detail.
- [pipeline-contracts.md](pipeline-contracts.md) - `Pipeline`/presets vs. the declarative engine.
- [testing.md](testing.md) - regression tests and the test layout.
- [../pipelines.md](../pipelines.md) - the declarative YAML/JSON pipeline spec format.
- [../migration/](../migration/) - calling `eitprocessing`/`resurfemg` directly vs. through `m3resp`.
