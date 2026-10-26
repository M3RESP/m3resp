# M3Resp Stage 2

Stage 2 turns `m3resp` from a thin wrapper around `eitprocessing`/`resurfemg` ([stage1.md](stage1.md)) into a shared multimodal data model and framework, without breaking anything Stage 1 already provided. This page lists the documentation pages of the stage 2 work. See {doc}`architecture` for the code as it is now.

## Concepts - what each object is

- [concepts/session.md](../concepts/session.md) - `M3Session`, its typed collections, and its full method list.
- [concepts/signals.md](../concepts/signals.md) - `Signal`/`TimeSeries`.
- [concepts/events-and-breaths.md](../concepts/events-and-breaths.md) - `Event`/`Interval`/`BreathEvent`, and `IntervalData`/`EventData`.
- [concepts/parameters.md](../concepts/parameters.md) - `ParameterResult`.
- [concepts/quality.md](../concepts/quality.md) - `QualityFlag`.
- [concepts/synchronization.md](../concepts/synchronization.md) - alignment, `LinkedBreath`, and breath timing parameters.
- [concepts/provenance.md](../concepts/provenance.md) - `ProvenanceRecord`, `ProcessingHistory`, and the persisted (Layer 2) data model.

## Tutorials - end-to-end walkthroughs

- {doc}`../tutorials/10_eit_only`
- {doc}`../tutorials/20_emg_only`
- {doc}`../how-to/export-results`

## Migration

- [migration.md](migration.md) - index and cross-modality bookkeeping.
- [migration/from-eitprocessing.md](migration/from-eitprocessing.md)
- [migration/from-resurfemg.md](migration/from-resurfemg.md)

## Developer reference

- [Architecture](architecture.md): the two data-model layers and the folders of the code.
- [Adapters](adapters.md): how M3Resp calls the two packages.
- [Preset contracts](preset-contracts.md): presets and workflow files.
- [Testing](testing.md): the comparison tests and the test folders.

## See also

- [stage1.md](stage1.md) - the Stage 1 wrapper layer these all build on.
- [workflows.md](../workflows.md) - the declarative YAML/JSON workflow spec format (its own deep reference, not duplicated here).
