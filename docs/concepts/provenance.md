# Provenance and processing history

## Plain-language overview

There are three records of "what produced this result," each heavier and
more structured than the last. You do not choose between the first two:
both are written for you, by different parts of the software, and you read
whichever answers your question. The third you switch on when you need it.

1. `ProvenanceRecord`, the lightest record: one entry per `M3Session`
   method call (action name, modality, parameters, timestamp), appended to
   `session.provenance`. Written whenever you call a session method,
   including by hand, outside any pipeline. Read it to see what was asked
   for.

2. `ProcessingStep`/`ProcessingHistory`, a step up in detail. Instead of
   just "an action happened," it records exactly which `input_keys` and
   `output_keys` (the dictionary/context keys the step read from and wrote
   to) were touched, plus a `status` (`"succeeded"`, `"failed"`, or
   `"cancelled"`) and the installed version of each optional upstream
   package (`resurfemg`, `eitprocessing`) the step depends on. This is
   filled in automatically by the declarative workflow engine (the system
   that runs pipelines described by YAML spec files) whenever a step runs
   through it, without any step function needing to remember to log
   anything itself. Read it to see what a pipeline actually did, and to
   reproduce a step: the package versions are recorded nowhere else, and
   the same operation with the same parameters can give a different answer
   after an upstream upgrade. Only the engine writes this, so a session
   method called by hand appears in `session.provenance` and not here.

3. `m3resp.datamodel`, an opt-in (only active if explicitly attached)
   heavier layer - "Layer 2" in the architecture diagram, where "Layer 1"
   is the runtime objects in `m3resp.data` (see
   [../stage2.md](../stage2.md)); that numbering is unrelated to this
   list. If you set `session.datamodel = DataModelRecorder(session)`, the
   session additionally builds up a set of validated, database-style
   records (`Case`, `RecordingSession`, `ProcessingRun`, `DerivedFeature`,
   `QualityAnnotation`, and more) inside a `DataModelStore`. This is meant
   for a proper audit trail (a complete, checkable record suitable for
   compliance/reproducibility purposes) and for a future GUI or backend
   service to query. Attach it when you intend to export a finished,
   checkable dataset; the first two records stay in memory on the session
   and are not validated. It is also a different shape of bookkeeping than
   `ProcessingHistory`: a full pipeline run becomes one `ProcessingRun`,
   and so does each individual session method call, versus one
   `ProcessingStep` per step inside a pipeline.

There is also a `validate_store(store, require_complete=True)` option that
checks the deeper layer has all the descriptive detail a finished dataset
needs (units, sampling rates, file checksums); a store built while a
session is still mid-run usually will not pass that check yet, since some
of those details (like a file's checksum) only exist once the file is
actually written to disk.

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
cardinality from `session.processing_history`: one `ProcessingRun` per
pipeline run and one per session method call, vs. one `ProcessingStep` per
step inside a pipeline.

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
