# Provenance and processing history

## Plain-language overview

There are three layers of "what produced this result," each heavier and
more structured than the last:

1. `ProvenanceRecord`, the lightest layer: one entry per `M3Session` method
   call (action name, modality, parameters, timestamp), appended to
   `session.provenance`.

2. `ProcessingStep`/`ProcessingHistory`, a step up in detail. Instead of
   just "an action happened," it records exactly which `input_keys` and
   `output_keys` (the dictionary/context keys the step read from and wrote
   to) were touched, plus a `status` (`"succeeded"`, `"failed"`, or
   `"cancelled"`). This is filled in automatically by the declarative
   workflow engine (the system that runs pipelines described by YAML spec
   files) whenever a step runs through it, without any step function
   needing to remember to log anything itself. It does not replace
   `ProvenanceRecord`, both exist side by side, tracking slightly
   different things.

3. Layer 2, `m3resp.datamodel`, an opt-in (only active if explicitly
   attached) heavier layer. If you set
   `session.datamodel = DataModelRecorder(session)`, the session
   additionally builds up a set of validated, database-style records
   (`Case`, `RecordingSession`, `ProcessingRun`, `DerivedFeature`,
   `QualityAnnotation`, and more) inside a `DataModelStore`. This is meant
   for a proper audit trail (a complete, checkable record suitable for
   compliance/reproducibility purposes) and for a future GUI or backend
   service to query. It is a different shape of bookkeeping than
   `ProcessingHistory`: for example, there is exactly one `ProcessingRun`
   record per full pipeline run, versus one `ProcessingStep` per individual
   step inside that run.

There is also a `validate_store(store, require_complete=True)` option that
checks the deeper layer has all the descriptive detail a finished dataset
needs (units, sampling rates, file checksums); a store built while a
session is still mid-run usually will not pass that check yet, since some
of those details (like a file's checksum) only exist once the file is
actually written to disk.

Three complementary layers record "what produced this result," from
lightest to heaviest.

## `ProvenanceRecord` (`m3resp.core.provenance`)

Stage 1's minimal log entry, appended to `session.provenance` by every
instrumented `M3Session` method (`session._record(action, modality, **parameters)`):

```python
@dataclass
class ProvenanceRecord:
    action: str
    modality: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ...   # UTC ISO 8601, set automatically
```

## `ProcessingStep`/`ProcessingHistory` (`m3resp.data.processing`)

The runtime counterpart with more structure: where `ProvenanceRecord` is
"action + modality + parameters," `ProcessingStep` additionally names the
context keys a step read and wrote, plus an outcome status:

```python
@dataclass
class ProcessingStep:
    name: str
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    software: str = "m3resp"
    version: str | None = None
    optional_package_versions: dict[str, str | None] = field(default_factory=dict)
    timestamp: str = ...
    status: str = "succeeded"   # "succeeded" | "failed" | "cancelled"
```

`session.processing_history` (a `ProcessingHistory`) is populated
automatically for every step run through the declarative
`m3resp.workflows` engine (see [../pipelines.md](../pipelines.md)) - the
engine knows the exact operation, bindings, parameters, timing, and outcome
of each step and records it after execution, so no step function has to
remember to call anything itself. `ProvenanceRecord` is not replaced by
this; both stay in use.

`name` (the registered operation id, e.g. `"eit.pixel_tiv"`) plus
`parameters` plus `input_keys`/`output_keys` are enough to replay a
deterministic step exactly - but only once you also know which version of
each optional upstream package (`resurfemg`, `eitprocessing`) it ran
against, since the same operation/parameters can produce different output
after a library upgrade. `optional_package_versions` records the installed version of every optional package the step's
operation declares a dependency on (`None` if that package wasn't
importable), captured by the engine at execution time.

## Layer 2: `m3resp.datamodel` (opt-in, persisted/audit entities)

Attach `session.datamodel = DataModelRecorder(session)` to additionally
record a validated, queryable audit trail (`Case`, `RecordingSession`,
`SignalStream`, `DataFile`, `ProcessingRun`, `DerivedFeature`,
`QualityAnnotation`, ...) into a `DataModelStore`. This is Layer 2 in the
architecture diagram (see [../developer/architecture.md](../developer/architecture.md)):
a validated record of what happened, built for later consumption by an
audit trail, export, or a future backend/GUI service layer - separate in
cardinality from `session.processing_history` (one `ProcessingRun` per
pipeline run, vs. one `ProcessingStep` per step).

```python
from m3resp.datamodel.recorder import DataModelRecorder

session.datamodel = DataModelRecorder(session)
# ... run pipelines/session methods as usual ...

from m3resp import export_store, validate_store
export_store(session.datamodel.store, "results/datamodel/")
validate_store(session.datamodel.store)   # reference checks only, by default
```

`validate_store(store, require_complete=True)` additionally checks that each
record carries the descriptive fields the full data model expects (units,
sampling rate, start time, file checksums) - useful before saving a
finished dataset, but a store recorded while a session is still running
usually does not pass this yet (e.g. a file isn't checksummed until it's
written to disk).
