# `ParameterResult`

## Plain-language overview

This is the type for any computed number (a "parameter" in the scientific
sense: tidal impedance variation, respiratory rate, EMG amplitude, and so
on), whether it is a single scalar number or an array (like a map of
regional lung ventilation).

Its fields let you say exactly what the number applies to, since these are
all optional and can be combined:

- `breath_id`, this number is about one specific breath.
- `breath_ids`, this number was computed across several breaths (for
  example a rolling average).
- `event_id`, this number is tied to a specific
  [`Event`](events-and-breaths.md) - for example a blood-gas value recorded
  at that timepoint, or a validation value resulting from a labeled
  intervention like a Baydur maneuver.
- `start_time`/`end_time`, this number applies to a time window (for
  example "during the intervention").
- If none of the above are set, the number applies to the whole recording.

Where they come from: same pattern as `Signal`, the adapters have
`to_parameters()` methods, called automatically by
`preprocess_eit`/`postprocess_emg`. There is also a cross-modality source:
`session.compute_multimodal_parameters()`, covered in
[synchronization.md](synchronization.md). All of them land in
`session.parameter_results`, which supports filtering like
`.for_modality("eit")` and exports to a CSV file.

A named, unit-tagged metric produced by a processing step - covers both
scalar metrics (EIT TIV, EMG amplitude, respiratory rate) and array-valued
ones (regional ventilation maps).

```python
@dataclass
class ParameterResult:
    name: str
    value: float | np.ndarray
    modality: str
    unit: str | None = None
    breath_id: str | None = None
    breath_ids: list[str] | None = None
    event_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    region: str | None = None
    channel: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

A parameter can be scoped to whichever of these apply (all are optional and
independent, so combinations - e.g. one metric per breath within a time
period - are possible):

- a single breath (`breath_id`);
- multiple breaths, e.g. a metric computed over a rolling window of breaths
  (`breath_ids`);
- a single timepoint (`start_time` set, `end_time` left `None`);
- a time period, e.g. during an intervention or one 30-second window
  (`start_time` and `end_time` both set) - `start_time`/`end_time` are single
  values on one `ParameterResult`, not a list, so a signal split into
  repeated 30-second windows would be one `ParameterResult` per window, not
  one `ParameterResult` holding all of them. (Nothing in the codebase
  produces periodic-window results like this yet - this is illustrating what
  the fields support, not existing behavior.)
- the whole signal, when none of the above are set.

`unit` is normalized via `m3resp.data.units.normalize_unit`. `is_scalar`
(`np.ndim(value) == 0`) and `to_dict()` (JSON-serializable, array values
become lists) are the two helper members.

## Where `ParameterResult`s come from

- Per-modality: `EITProcessingAdapter.to_parameters`/
  `ReSurfEMGAdapter.to_parameters` convert the adapters' preprocessing
  output into `ParameterResult`s (see
  [../developer/adapters.md](../developer/adapters.md)); `preprocess_eit`/
  `postprocess_emg` call these and add the results to
  `session.parameter_results`.
- Cross-modality: `session.compute_multimodal_parameters()` computes timing
  delays, breath-duration differences, and event-agreement scores from
  `session.linked_breaths` - see [synchronization.md](synchronization.md).
  These are deliberately timing-only metrics. A cross-modality index that
  jointly analyzes signal *values* rather than breath timing (e.g. an
  EMG-effort-to-EIT-pendelluft coupling index) is genuinely new science with
  no upstream equivalent, which is out of scope for Stage 2 - see
  ["A completely new algorithm with no upstream equivalent"](../developer/architecture.md)
  and the Stage 3 outlook there for where it belongs once Stage 3's native
  packages exist.

`session.parameter_results` (`m3resp.data.collections.ParameterResultCollection`)
is queryable via `.for_modality(name)`/`.for_name(name)`, and exports to
`parameter_results.csv` via `session.export_summary()` - see
[../tutorials/export-results.md](../tutorials/export-results.md).
