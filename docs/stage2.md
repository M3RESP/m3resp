# M3Resp Stage 2

Stage 2 turns `m3resp` from a thin wrapper around `eitprocessing`/`resurfemg`
([stage1.md](stage1.md)) into a shared multimodal data model and framework,
without breaking anything Stage 1 already provided. This page is a short
index; the full documentation lives in the directories linked below (see
[`plan/stage2/stage2_concrete_plan.md`](../plan/stage2/stage2_concrete_plan.md)
Sec 23 for the structure these follow).

## Concepts - what each object is

- [concepts/session.md](concepts/session.md) - `M3Session`, its typed collections, and its full method list.
- [concepts/signals.md](concepts/signals.md) - `Signal`/`TimeSeries`.
- [concepts/events-and-breaths.md](concepts/events-and-breaths.md) - `Event`/`BreathEvent`.
- [concepts/parameters.md](concepts/parameters.md) - `ParameterResult`.
- [concepts/quality.md](concepts/quality.md) - `QualityFlag`.
- [concepts/synchronization.md](concepts/synchronization.md) - alignment, `LinkedBreath`, and multimodal parameters.
- [concepts/provenance.md](concepts/provenance.md) - `ProvenanceRecord`, `ProcessingHistory`, and the persisted (Layer 2) data model.

## Tutorials - end-to-end walkthroughs

- [tutorials/eit-only.md](tutorials/eit-only.md)
- [tutorials/emg-only.md](tutorials/emg-only.md)
- [tutorials/multimodal-eit-emg.md](tutorials/multimodal-eit-emg.md)
- [tutorials/export-results.md](tutorials/export-results.md)

## Migration

- [migration.md](migration.md) - index and cross-modality bookkeeping.
- [migration/from-eitprocessing.md](migration/from-eitprocessing.md)
- [migration/from-resurfemg.md](migration/from-resurfemg.md)

## Developer reference

- [developer/architecture.md](developer/architecture.md) - the two data-model layers and the package map.
- [developer/adapters.md](developer/adapters.md) - the adapter conversion boundary.
- [developer/pipeline-contracts.md](developer/pipeline-contracts.md) - `Pipeline`/presets vs. the declarative engine.
- [developer/testing.md](developer/testing.md) - regression tests and the test layout.

## See also

- [stage1.md](stage1.md) - the Stage 1 wrapper layer these all build on.
- [pipelines.md](pipelines.md) - the declarative YAML/JSON pipeline spec format (its own deep reference, not duplicated here).
