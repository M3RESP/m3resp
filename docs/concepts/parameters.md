# `ParameterResult`

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
- a time period, e.g. during an intervention or every 30 seconds
  (`start_time` and `end_time` both set);
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

`session.parameter_results` (`m3resp.data.collections.ParameterResultCollection`)
is queryable via `.for_modality(name)`/`.for_name(name)`, and exports to
`parameter_results.csv` via `session.export_summary()` - see
[../tutorials/export-results.md](../tutorials/export-results.md).
