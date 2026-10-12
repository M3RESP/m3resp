# `QualityFlag`

## Plain-language overview

This represents the result of one quality check: did this signal/breath
pass or fail some validity test? It's the same idea as
[`ParameterResult`](parameters.md) - a value tied to a signal/breath/time
window - but for a pass/fail check instead of a measurement. Key fields:

- `passed`, a true/false verdict.
- `severity`, how serious a failure is: `"info"`, `"warning"`, `"error"`,
  or `"critical"`.
- `threshold`/`value`, the cutoff used and the actual measured value, so
  you can see why it passed or failed.

An important rule: a check that does not apply (skipped, not computable)
emits no flag. Every flag records a real pass or fail, so "we did not check
this" never shows up as "this passed".

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
    category: str | None = None
    signal_name: str | None = None
    breath_id: str | None = None
    breath_ids: list[str] | None = None
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

A flag about one breath sets `breath_id`. A flag about several breaths
judged together sets `breath_ids`, for example ten breaths marked as a
stable stretch, or a set of breaths with gaps between them. A flag
sets one or the other; setting both, or an empty `breath_ids` list,
raises `ValueError`. The per-breath steps use the breath's position in its
breath list as the id (`"0"`, `"1"`, ...).

```python
QualityFlag(
    name="stable_stretch",
    passed=True,
    severity="info",
    modality="eit",
    breath_ids=["3", "4", "5", "6"],
)
```

In `quality_flags.csv`, `breath_ids` is written as a Python list
(`['3', '4', '5', '6']`); `ast.literal_eval` reads it back. In
`summary.json` it is a JSON list.

`QualityFlag` mirrors the persisted `QualityAnnotation` entity
(`m3resp.datamodel.entities`, Layer 2 - see
[provenance.md](provenance.md)) but is the lightweight, in-memory object a
quality check actually produces during a workflow run; conversion to
`QualityAnnotation` happens in `DataModelRecorder`. Each flag becomes one
annotation that keeps the flag's time window (`start_time`/`end_time`, in
seconds) and its breaths in `QualityAnnotation.breath_ids`: a one-item list
for a flag about one breath, the full list for a flag about several
breaths. A skipped or not-applicable check emits no flag; the native EMG
quality steps follow this rule.

## Where `QualityFlag`s come from

`EITProcessingAdapter.to_quality_flags`/`ReSurfEMGAdapter.to_quality_flags`
convert adapter preprocessing output into `QualityFlag`s (see
[../developer/adapters.md](../developer/adapters.md)); `preprocess_eit`/
`postprocess_emg` call these and add the results to `session.quality`
(`m3resp.data.collections.QualityReport`).

`session.quality` is queryable via `.failed()`, `.for_modality(name)`,
`.for_signal(signal_name)`, and exports to `quality_flags.csv` via
`session.export_summary()`.
