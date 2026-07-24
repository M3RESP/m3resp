# `QualityFlag`

## Plain-language overview

This represents the result of one quality check: did this signal/breath
pass or fail some validity test? Key fields:

- `passed`, a true/false verdict.
- `severity`, how serious a failure is: `"info"`, `"warning"`, `"error"`,
  or `"critical"`.
- `threshold`/`value`, the cutoff used and the actual measured value, so
  you can see why it passed or failed.

An important rule: if a check genuinely does not apply (skipped, not
computable), the code should simply not emit a flag at all, rather than
inventing a fake pass/fail. This avoids quietly turning "we did not check
this" into "this passed," which would be misleading.

`QualityFlag` is the lightweight, in-memory version of a more permanent
database-style record called `QualityAnnotation` (see
[provenance.md](provenance.md)); the conversion between the two only
happens if you opt into the deeper persistence layer (Layer 2, described
in that document).

The outcome of one quality check against a signal, breath, or run.

```python
@dataclass
class QualityFlag:
    name: str
    passed: bool
    severity: Severity        # "info" | "warning" | "error" | "critical"
    modality: str | None = None
    signal_name: str | None = None
    breath_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    message: str | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

A flag covers the whole signal by default. To represent a signal that's only
bad for part of its duration (e.g. good except for a noisy period in the
middle), emit one `QualityFlag` per affected window with `start_time`/
`end_time` set, rather than a single flag for the whole signal -
`signal_name` ties them all back to the same signal.

`QualityFlag` mirrors the persisted `QualityAnnotation` entity
(`m3resp.datamodel.entities`, Layer 2 - see
[provenance.md](provenance.md)) but is the lightweight, in-memory object a
quality check actually produces during a pipeline run; conversion to
`QualityAnnotation` happens at the `DataModelRecorder` boundary, not here. A
skipped/not-applicable check should simply not emit a flag, rather than
emitting one with an invented "passed" or "failed" verdict - this is what
the native EMG quality steps already do.

## Where `QualityFlag`s come from

`EITProcessingAdapter.to_quality_flags`/`ReSurfEMGAdapter.to_quality_flags`
convert adapter preprocessing output into `QualityFlag`s (see
[../developer/adapters.md](../developer/adapters.md)); `preprocess_eit`/
`postprocess_emg` call these and add the results to `session.quality`
(`m3resp.data.collections.QualityReport`).

`session.quality` is queryable via `.failed()`, `.for_modality(name)`,
`.for_signal(signal_name)`, and exports to `quality_flags.csv` via
`session.export_summary()`.
