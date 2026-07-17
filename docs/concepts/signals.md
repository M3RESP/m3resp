# `Signal` and `TimeSeries`

`m3resp.data.timeseries.TimeSeries` is the base runtime type every
continuous signal in `m3resp` is represented as, whatever its modality: a
value array paired with its time axis, plus sampling rate, unit, and
free-form metadata.

```python
@dataclass
class TimeSeries:
    values: np.ndarray
    time: np.ndarray
    sample_frequency: float | None = None
    unit: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`values`'s leading axis must line up with `time` (one entry per timepoint),
but there's no constraint on its remaining dimensions - a 1D array (one
scalar per timepoint, e.g. global impedance), a 2D array (one value per
timepoint per channel/region), or higher-dimensional arrays (e.g. EIT
pixel-impedance frames: time × rows × cols) are all valid. `unit` is
normalized via `m3resp.data.units.normalize_unit` (e.g. `"uV"` -> `"µV"`),
so equivalent units from different loaders end up comparable instead of
drifting into incompatible spellings.

`m3resp.data.signals.Signal` subclasses `TimeSeries`, adding modality and
provenance context:

```python
@dataclass
class Signal(TimeSeries):
    modality: Modality = "unknown"        # "eit" | "emg" | "ventilator" | "pressure" | "flow" | "unknown" | any str
    channel: str | None = None
    source: str | None = None
    processing_state: ProcessingState = "raw"   # "raw" | "filtered" | "processed" | "derived"
    derived_from: ProcessingState | None = None
    method: str | None = None
```

`modality` is an open vocabulary, not an enum - a loader/adapter can use any
string (e.g. a new device, or a more specific quantity type than
"ventilator" such as pressure/flow/volume/EAdi).

`processing_state`:

- `"raw"` - untouched, straight from the source/loader.
- `"filtered"` - an intermediate signal-processing step has been applied;
  not yet the final signal a downstream parameter computation should use.
- `"processed"` - the final signal for this channel, ready to compute
  parameters/results from.
- `"derived"` - computed from another signal (e.g. a difference between two
  signals), rather than a step in that signal's own raw -> filtered ->
  processed pipeline; `derived_from` records which state it was derived
  from.

Multiple differently produced signals can share the same `channel` and
`processing_state` (e.g. two "filtered" variants using different filter
methods) - use `method` to tell them apart, not a new state.

## Where `Signal`s come from

Adapters (`EITProcessingAdapter.to_signals`/`ReSurfEMGAdapter.to_signals`,
see [../developer/adapters.md](../developer/adapters.md)) convert whatever
`eitprocessing`/`resurfemg` return into `Signal` instances at the public
boundary - everything downstream (session storage, pipeline steps, export)
operates on this type instead of vendor-specific objects.
`session.preprocess_eit()`/`session.preprocess_emg()` call this conversion
by default and add the results to `session.signals`
(`m3resp.data.collections.SignalCollection`).
