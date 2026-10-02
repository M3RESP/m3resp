# Events, intervals and breaths

## Plain-language overview

`m3resp.data.events` defines three types for things that happen in time.
All modalities share them.

| Type | What it is | Example |
|---|---|---|
| `Event` | Something that happens at one instant: one `time` | a heartbeat, a blood-gas draw |
| `Interval` | Something that lasts: a `start_time` and an `end_time` | an occlusion, a period of noise, an intervention |
| `BreathEvent` | One breath: an `Interval` with a turning point (`extremum_time`) inside it | a breath found in EIT, EMG or ventilator data |

Every `BreathEvent` is also an `Interval`, so anything written for intervals
works on breaths too. EIT, EMG and ventilator breath detection all produce
the same `BreathEvent` type, instead of each modality having its own breath
class. That is what lets breaths from different modalities be compared and
matched later (see [synchronization.md](synchronization.md)).

When a result has **one value per breath, interval or event**, keep the
value next to the time it belongs to with `IntervalData` or `EventData`; see
[Values per interval or event](#values-per-interval-or-event).

Breaths are not kept in a container class of their own. They live inside
`session.events`, a plain dictionary, under keys like `"eit_breaths"`; see
[Where breath/event lists live](#where-breathevent-lists-live).

## `Event`

Something that happens at a single instant, from one modality:

```python
@dataclass
class Event:
    name: str
    modality: str
    time: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    sample_index: int | None = None
    signal_name: str | None = None
    sample_frequency: float | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`name` says what kind of event it is (`"blood_gas_draw"`); `label` names one
particular occurrence (`"baydur_maneuver"`).

`sample_index` is only meaningful together with `signal_name`/
`sample_frequency`, which say which signal and time axis it's relative to -
a modality can have multiple signals at different sampling rates, so
`sample_index` alone doesn't tell you which one. `Interval` has the same
fields for the same reason. All three are `None` when an event wasn't
derived from indexing into a specific signal.

### `id`

`id` is generated automatically (an in-memory identifier, not persisted or
globally unique like Layer 2's ids) so other Layer 1 objects can reference
this exact event, e.g. `ParameterResult.event_id`. It's excluded from
equality so two structurally identical events still compare equal.
`Interval.id` and `BreathEvent.id` work the same way.

## `Interval`

Something that lasts, from one modality:

```python
@dataclass
class Interval:
    modality: str
    start_time: float
    end_time: float
    _: KW_ONLY             # everything below is given by name
    name: str              # required: what kind of interval, e.g. "occlusion"
    id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    label: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    sample_frequency: float | None = None
    signal_name: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
```

```python
occlusion = Interval("ventilator", 10.0, 12.5, name="occlusion")
occlusion.duration  # 2.5
```

Only `modality`, `start_time` and `end_time` may be given by position;
everything else is given by name. An interval whose `end_time` is before its
`start_time` raises `ValueError`.

`start_time`/`end_time` are always the authoritative, real-world times, in
seconds - they don't need to be recomputed from an index. The `*_index`
fields are optional sample positions in the signal the interval was found
in; `sample_frequency` and `signal_name` say which time axis those positions
are relative to, since different signals have different start times,
durations, and sampling rates.

`coerce_interval` turns other kinds of input into an `Interval`: a
dictionary, an object with `start_time`/`end_time` (such as eitprocessing's
`Interval`), or a `(start_time, end_time)` pair. Pass `name=` and
`modality=` for whatever the input does not carry. An input that is a breath
(named `"breath"`, or carrying a turning point) comes back as a
`BreathEvent`, so the turning point is kept; an input with a turning point
but another name raises `ValueError`, because a plain `Interval` has no
place for it.

## `BreathEvent`

One breath. It has every field of `Interval`, with `name` always
`"breath"` (it cannot be set to anything else), plus two of its own:

```python
@dataclass(kw_only=True)
class BreathEvent(Interval):
    name: str = field(default="breath", init=False)
    extremum_time: float | None = None
    extremum_index: int | None = None
```

```python
breath = BreathEvent("eit", 1.0, 2.0, extremum_time=1.5)
isinstance(breath, Interval)  # True
```

`extremum_time` is the moment the signal turns from inhalation to exhalation;
`extremum_index` is its sample position. Both are `None` when the detector
didn't report one. It is called an extremum, not a peak, because the signal
can turn at a maximum (impedance, volume, EMG envelope) or at a minimum
(esophageal pressure, the deepest pressure of an occlusion).
`coerce_breath_event` reads the turning point from the first of
`extremum_time`/`extremum_index`, `peak_time`/`peak_index` (older m3resp
versions and other detectors; a warning names the new keys) and
`middle_time` (eitprocessing). The time and its position always come from
the same pair.

A `BreathEvent` and an `Interval` with the same times are not equal: one
says "this was a breath", the other does not. `coerce_breath_event` refuses
an interval named anything other than `"breath"`, so an occlusion or a
period of noise cannot be counted as a breath by mistake.

An occlusion (Pocc) is stored as a `BreathEvent`, with
`metadata["event_type"] == "pocc"`: it is an effort to breathe in against a
closed airway, and its deepest pressure is a turning point that a plain
`Interval` could not keep.

## Values per interval or event

Some results have one value per breath, per occlusion or per heartbeat
rather than one value per sample: the tidal impedance variation of each
breath, a pixel map of breath timing for each breath, or the amplitude of
each ECG peak. `IntervalData` and `EventData` (in `m3resp.data.event_data`)
keep each value next to the interval or event it belongs to, so the timing
is never lost. Inside m3resp they take the place of eitprocessing's
`IntervalData` and `SparseData`.

```python
from m3resp import IntervalData

tiv = IntervalData(
    name="tidal_impedance_variation",
    modality="eit",
    intervals=session.get_events("eit_breaths"),
    values=[1.5, 1.7, 1.6],
    unit="AU",
)
```

| Field | Meaning |
|---|---|
| `name` | What the values are |
| `modality` | Which device the values were measured with. This can differ from the device the intervals came from: EIT values can be computed over breaths found in ventilator data. |
| `intervals` (`IntervalData`) / `events` (`EventData`) | The intervals or events, in order. A list of breaths works as intervals. |
| `values` | One value per interval or event, in the same order: a list, a tuple, or an array whose first axis runs over the intervals. A value can be a number or an array, such as a pixel map. `None` when the intervals themselves are the result. A dictionary or a single number is refused. |
| `category`, `unit` | Physical quantity and unit, tidied the same way as on `ParameterResult` |
| `method`, `metadata` | Which method produced the values, and any extra information |

A `values` list that does not have exactly one entry per interval or event
raises `ValueError`, so a value can never end up next to the wrong breath.
`to_dict()` gives the same content as plain lists and dictionaries, ready to
write to a JSON file.

Steps store their `IntervalData` in `session.interval_data`. The EIT steps
that do this:

| Step | Name | One value per breath |
|---|---|---|
| `eit.continuous_tiv` | `continuous_tivs` | TIV |
| `eit.eeli` | `continuous_eelis` | EELI |
| `eit.pixel_tiv` | `pixel_tivs` | a `PixelMap` of TIV (see [pixel-maps.md](pixel-maps.md)) |
| `eit.pixel_breaths` | `pixel_breaths` | each pixel's breath start, middle and end time |

The breaths are the ones eitprocessing computed the values over, found on the
impedance waveform (global or regional) the step was given. Steps that use the
same breath detector and signal share the very same breath objects, and a
breath that is also in `session.events["eit_breaths"]` (same start and end
time) is that stored breath. `preprocess_eit()` and `detect_eit_breaths()` do
the same, in either order. The rule lives in
`m3resp.data.events.reuse_matching_breaths`. Their times are on the EIT
recording's own clock, like the breaths in `session.events`. A NaN value (for
example the TIV of a breath that could not be measured) stays next to its
breath; it is not dropped.

## Where breath/event lists live

There is deliberately no `BreathCollection` type: breaths live in
`session.events`, a `dict[str, list[BreathEvent]]` populated by
`session.detect_eit_breaths()` (`"eit_breaths"`),
`session.detect_emg_breaths()` (`"emg_breaths"`), and any ventilator breaths
normalized during `postprocess_emg` (`"ventilator_breaths"`). Use
`session.add_events(name, events)`/`session.get_events(name)` for direct
access. This predates the typed-collection work and is depended on
throughout Stage 1 - introducing a second container would fork that API
rather than reconcile with it.

Moving breaths, intervals and events onto a common clock is done by
`align_events_by_modality_offset`, which shifts all three types.

To match breaths across modalities into a single physiological event, see
[`LinkedBreath`](synchronization.md).
