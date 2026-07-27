# `Signal` and `TimeSeries`

## Plain-language overview

`TimeSeries` is the base building block for anything that changes over time
and was recorded continuously (as opposed to a single instantaneous event).
It is a dataclass (a Python shortcut for defining a simple object that just
holds a fixed set of fields):

- `values`, the actual numbers, stored as a numpy array (a fast,
  multi-dimensional grid of numbers used throughout scientific Python).
- `time`, the timestamp for each value, so you know when each number was
  recorded.
- `sample_frequency`, how many measurements were taken per second.
- `unit`, what the numbers mean physically (for example `"uV"` for
  microvolts). This gets normalized (converted to one consistent spelling)
  so `"uV"` from one device and `"uV"` written differently by another both
  end up as the same thing, and can safely be compared.

`values` does not have to be a simple list of numbers. It can be 1D (one
number per moment in time, like total lung impedance), 2D (one number per
moment per channel, like one column per electrode), or higher-dimensional
(an EIT frame is a 2D image at every moment in time, so it is 3D overall:
time by rows by columns).

`Signal` builds on top of `TimeSeries` (in Python this is called
subclassing: `Signal` gets everything `TimeSeries` has, plus extra fields)
by adding:

- `modality`, which **device or technique** recorded this (`"eit"`, `"emg"`,
  `"ventilator"`, or any custom string).
- `category`, what the numbers **physically are** (`"impedance"`,
  `"airway_pressure"`, `"airflow"`, `"volume"`, ...).
- `processing_state`, where this signal sits in its journey from raw to
  usable: `"raw"` means straight off the device, untouched; `"filtered"`
  means cleaned up a bit but not the final version; `"processed"` means the
  final, ready-to-use version; `"derived"` means computed from another
  signal (like a difference between two signals), rather than being a
  cleanup step of its own raw data.
- `method`, which distinguishes two signals that are both, say,
  `"filtered"` but used different filtering techniques.

Where they come from: the adapters (the wrapper objects that translate
calls from `m3resp` into calls on an outside library) have `to_signals()`
methods that convert whatever the outside library returns into these
standard `Signal` objects. `session.preprocess_eit()`/`preprocess_emg()`
call that conversion automatically and add the results to `session.signals`.

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
    modality: Modality = "unknown"        # "eit" | "emg" | "ventilator" | "monitor" | "unknown" | any str
    category: Category | None = None      # "impedance" | "airway_pressure" | "airflow" | "volume" | ... | any str
    channel: str | None = None
    source: str | None = None
    processing_state: ProcessingState = "raw"   # "raw" | "filtered" | "processed" | "derived"
    derived_from: ProcessingState | None = None
    method: str | None = None
```

`method` names the algorithm that produced the signal, formatted
`"<library>.<function_or_class_name>"` (e.g. `"resurfemg.moving_baseline"`,
`"eitprocessing.RateDetection"`). The prefix disambiguates functions that
share a name across libraries; it is convention only - a plain string with no
runtime validation - so it's up to each adapter/step to follow it rather than
passing through an upstream library's own unqualified names.

### `modality` vs `category`: two independent axes

`modality` is the **device/technique**; `category` is the **physical
quantity**. They are deliberately separate fields because they do not nest:

- one device emits several quantities (a ventilator produces pressure *and*
  flow *and* volume);
- one quantity comes from several devices (airway pressure can come from a
  ventilator or a standalone monitor).

Collapsing them into a single string makes some combinations inexpressible.
Ventilator volume is the concrete case: with only `modality`, a volume channel
had to be tagged either `"ventilator"` (losing which quantity it was) or
`"volume"` (losing which device produced it). The persisted Layer 2 model has
always kept the two apart - `Device.device_type` and `SignalStream.signal_type`
- so a single Layer 1 string had to be split heuristically on the way in, and
`"ventilator_volume"` was unreachable in practice.

`modality`'s vocabulary lines up 1:1 with Layer 2's `Device.device_type`.
`category`'s vocabulary is deliberately *modality-agnostic*, following the same
principle as `eitprocessing`'s shared category catalogue: a taxonomy of
physical quantities with no notion of which device measured them.

Both are open vocabularies, not enums - a loader/adapter can use any string.
A known alias is canonicalized on construction (`category="paw"` is stored as
`"airway_pressure"`); an unrecognized value is kept verbatim, so a custom or
experimental category is never silently dropped or relabelled.

Query either axis independently:

```python
session.signals.for_modality("ventilator")      # every ventilator channel
session.signals.for_category("airway_pressure") # pressure, whatever recorded it
```

`for_category` is available on `session.signals`, `session.parameter_results`,
and `session.quality`, and accepts aliases (`for_category("paw")` works).

The category vocabulary is extensible without editing the package - see
`m3resp.data.categories.register_category_alias` for a single addition, and
`load_category_aliases` to adopt an externally-maintained catalogue from a
YAML/JSON file.

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
