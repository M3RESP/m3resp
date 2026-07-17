# Provenance and processing history

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
