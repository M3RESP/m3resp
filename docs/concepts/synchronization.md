# Synchronization and multimodal parameters

## Plain-language overview

This module's job is lining up data from different modalities (EIT/EMG/
ventilator) that were recorded on separate clocks or files, so they can be
compared on one shared time axis. Key pieces:

- `session.synchronize_raw_modalities(...)` gives each recording a start
  time on one shared clock, using a manual offset you supply (for example
  "EMG started 5 seconds after EIT"). The offset is a single number in
  seconds that you provide; the package applies it but never measures it
  (`estimate_sync_offset` only supports `method="manual"`). How you arrive
  at that number is up to you, and filtered signals are fine for the job.
  No samples are removed: a three-hour pressure recording stays three hours
  long beside a thirty-minute EIT recording that started 90 minutes in. See
  "Start times" below.
- `session.synchronize_multimodal_breaths(...)` does the same thing but for
  already-detected events (breaths), not raw signals.
- `resample_signal(...)` is a standalone utility that changes a signal's
  sampling rate to match another signal's, for the cases where you need two
  signals on one shared sample grid (e.g. a sample-by-sample comparison).
  It's not part of the alignment pipeline above and isn't called
  automatically: breath linking and the multimodal parameter calculations
  below work on real-world timestamps (`BreathEvent.start_time`/`end_time`/
  `peak_time`), not sample indices, so most analysis stays at each
  modality's original sample rate and resampling is only needed when you
  explicitly ask for it.

`LinkedBreath` is the object that represents "the same physical breath, as
seen by different modalities." It is a dictionary-like structure
(`breaths: dict[str, BreathEvent]`) mapping a modality name to that
modality's version of the matched breath, plus a `confidence` score based
on how close in time the matches were. Matching is done by
`link_breaths_by_time` (or `session.link_breaths()`), which works greedily
and one-to-one (it matches the closest available pair first, and each
breath can only be used once) with no clock-drift correction (it assumes
the recordings' clocks do not slowly drift apart over time, that is
explicitly out of scope). If a breath from one modality has no match in
the others, it still produces a `LinkedBreath` with only its own slot
filled in, so nothing gets silently dropped.

Once you have linked breaths, `session.compute_multimodal_parameters()`
turns them into cross-modality `ParameterResult`s using three underlying
calculations: timing delay (how many seconds apart two modalities' breath
timings are), breath duration difference (how much longer/shorter one
modality's breath looks compared to another's), and event agreement (what
fraction of breaths were detected consistently across all requested
modalities, a rough "did every sensor agree a breath happened here" score).

`m3resp.synchronization` (Milestone 2.5) aligns and links data across
modalities, deliberately kept modest: manual offset, timestamp alignment,
resampling, and nearest-neighbor breath linking. Clock-drift correction is
intentionally out of scope.

## Start times

`session.synchronize_raw_modalities(offset_seconds=..., reference_modality=...)`
stores one number per recording in `session.start_times`: the time, in
seconds on the shared clock, at which that recording's first sample was
taken. The reference recording starts at 0; a negative value means a
recording started earlier than the reference, a positive value later.
Calling it again replaces the start times.

Each modality's own results stay on that recording's own clock, so
`session.events["emg_breaths"]` still gives times from the start of the EMG
file. The start times are added only where modalities are compared -
`synchronize_multimodal_breaths`, `link_breaths` and
`plot_synchronization_comparison`:

```text
time on the shared clock = own time - own first time + start time
```

"Own first time" is 0 s for EMG and ventilator data. EIT files carry the
time of day instead (a Draeger file can start at 36528.6 s), so the EIT's
first time value is taken off first. A ventilator recording that came inside
the EIT or EMG file uses that modality's start time.

Standalone ventilator recordings (loaded with `source="ventilator"`) share
the `"ventilator"` start time. When two of them started at different moments
- a ventilator export and a separate monitor export, say - give one its own
start time with `"ventilator:<name>"`, the name it was loaded under:

```python
session.load_ventilator("ventilator.txt", source="ventilator")
session.load_ventilator("monitor.csv", source="ventilator", name="monitor")
session.synchronize_raw_modalities(
    offset_seconds={"eit": 0.0, "ventilator": -30.0, "ventilator:monitor": 12.5},
    reference_modality="eit",
)
```

A recording with its own entry uses it; the others use `"ventilator"`. A
name that is not loaded, or one for ventilator data inside the EIT or EMG
file, is refused.

Cutting a recording (`slice_emg`, `slice_eit`, `slice_ventilator`) moves its
start time later by the part cut off the front, so it stays lined up with the
other recordings. See [Cutting data to a time window](slicing.md).

## Aligning raw signals and events

- `session.synchronize_raw_modalities(...)` sets each recording's start time
  (see above). Run it before or after per-modality processing; it changes no
  samples.
- `session.synchronize_multimodal_breaths(method="manual_offset", offset_seconds=..., reference_modality=...)`
  shifts already-detected event lists (`session.events`) onto a common time
  axis. `offset_seconds` accepts either a single float or a per-modality
  mapping (e.g. `{"emg": 5.0}`).
- `m3resp.compute_offsets_from_timestamps(reference_modality, timestamps)` /
  `m3resp.align_events_by_modality_offset(events, offsets)` are the
  lower-level functions these session methods call.
- `m3resp.resample_signal(signal, target_frequency_hz)` resamples a `Signal`
  onto a common sampling rate.

## Linking breaths: `LinkedBreath`

`m3resp.link_breaths_by_time(breaths_by_modality, time_tolerance=0.5)` (or
`session.link_breaths(time_tolerance=0.5)`) matches `BreathEvent`s across
any number of modalities by how close their representative times are
(`peak_time`, falling back to the start/end midpoint) - greedily and
one-to-one, no clock-drift correction. A breath with no match in the other
modalities still produces a result with only its own slot filled, so no
input breath is silently dropped.

```python
@dataclass
class LinkedBreath:
    breaths: dict[str, BreathEvent] = field(default_factory=dict)
    time_tolerance: float = 0.5
    confidence: float | None = None       # 1 - (time_diff / time_tolerance), or None if unmatched
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[str]:
        """Which modalities contributed a breath, e.g. `["eit", "emg"]`."""
```

`breaths` maps an arbitrary modality name (`"eit"`, `"emg"`, `"ventilator"`,
or anything else) to that modality's matched breath. Run
`session.synchronize_multimodal_breaths(...)` first if the modalities are not already on a
common time axis - `session.link_breaths()` prefers the aligned event lists
over the raw ones when both exist.

## Multimodal parameters

`session.compute_multimodal_parameters(...)` (`m3resp.synchronization.multimodal_parameters`)
turns `session.linked_breaths` into cross-modality [`ParameterResult`](parameters.md)s
(plan Sec 21):

```python
session.link_breaths(time_tolerance=0.5)
results = session.compute_multimodal_parameters()
```

Three primitives, usable standalone on any `LinkedBreath`/`list[LinkedBreath]`:

- `compute_timing_delay(linked, from_modality, to_modality, anchor="start")` -
  signed delay in seconds between two modalities' breath anchors
  (`anchor` is `"start"`, `"peak"`, or `"end"`); `None` if either modality
  is missing from the link. Positive means `to_modality` occurs later.
- `compute_breath_duration_difference(linked, modality_a, modality_b)` -
  `duration(modality_a) - duration(modality_b)` in seconds; `None` if either
  modality is missing.
- `compute_event_agreement(linked_breaths, modalities)` - fraction of
  `linked_breaths` where every requested modality contributed a breath, a
  coarse breath-to-breath timing agreement score.

`compute_multimodal_parameters(linked_breaths, delay_pairs=None, duration_pairs=None, anchor="start")`
combines all three into `ParameterResult`s (`modality="multimodal"`, one
result per breath per pair, plus one aggregate event-agreement result per
delay pair). `delay_pairs`/`duration_pairs` default to every unordered pair
of modalities actually observed across `linked_breaths`, so a session that
only linked EIT and EMG never gets a meaningless ventilator pairing. A
breath missing either side of a pair is skipped for that pair rather than
raising, so a partially-linked recording still yields parameters for the
breaths that do have both modalities.

`session.compute_multimodal_parameters()` adds its results to
`session.parameter_results` (so they export to `parameter_results.csv`
alongside per-modality parameters) and records a provenance entry.

See [tutorials/multimodal-eit-emg.md](../tutorials/multimodal-eit-emg.md)
for an end-to-end walkthrough.
