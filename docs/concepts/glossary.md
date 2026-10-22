# Glossary

This page lists the words that the M3Resp documentation uses. Each word has one meaning on all pages. Names from the code, such as `BreathEvent`, are shown in `code` style.

```{glossary}
:sorted:

baseline
  The slowly changing lower level of the EMG {term}`envelope`. The step `emg.moving_baseline` calculates it as the 33rd percentile of the envelope in a moving window. The window is 7.5 s long and moves in steps of 0.2 s.

breath
  One breath that a detection step found in one {term}`signal`. M3Resp keeps each breath as a `BreathEvent`, with a start time, an {term}`extremum` time and an end time in seconds. EIT, EMG and ventilator detection all give `BreathEvent` objects, so you can compare breaths across modalities. Each breath is also an {term}`interval`.

channel
  One input of a measuring device. Examples are one EMG electrode pair, or the airway pressure input of a Biopac system. One file can hold more than one channel.

ECG
  Electrocardiography: the electrical activity of the heart. The ECG is also present in the EMG of the respiratory muscles. M3Resp finds the {term}`R peak` of each heartbeat and removes the ECG from the EMG.

EIT
  Electrical impedance tomography. EIT measures changes in the electrical impedance of the thorax and shows them as images of the lungs. It is one of the three {term}`modalities <modality>`.

EMG
  Electromyography: the electrical activity of a muscle. In M3Resp, EMG comes from the respiratory muscles, for example the diaphragm. It is one of the three {term}`modalities <modality>`.

envelope
  A smooth curve that follows the amplitude of the EMG. M3Resp calculates it from the filtered EMG over a moving window. The method is the RMS (root mean square), the ARV (average rectified value) or the median of the absolute signal. Breath detection on EMG uses the envelope.

event
  Something that happens at one moment, with one `time` in seconds. An example is a blood-gas draw. In code, an event is an `Event`.

extremum
  The turning point of one {term}`breath` in a signal. For EIT and for ventilator volume, it is the change from inspiration to expiration. For EMG, it is the highest point of the envelope. For an occluded breath, it is the lowest airway pressure. A breath keeps it as `extremum_time` in seconds and `extremum_index` as a sample number.

interval
  Something that lasts, with a `start_time` and an `end_time` in seconds. Examples are an occlusion and a period of noise. In code, an interval is an `Interval`.

linked breath
  One {term}`breath` that M3Resp matched across two or more modalities. Two breaths match when their extremum times are 0.5 s or less apart (you can change this tolerance). In code, a linked breath is a `LinkedBreath`.

modality
  The measuring method of a {term}`recording`. M3Resp has three modalities: {term}`EIT`, {term}`EMG` and {term}`ventilator`. Their names in code are `"eit"`, `"emg"` and `"ventilator"`.

offset
  The time in seconds between the clocks of two recordings. You measure or know the offset, and you give it to M3Resp. {term}`Synchronization` uses it to calculate the {term}`start time` of each recording.

parameter
  A number that M3Resp calculates from the signals, with its unit. A parameter can belong to one breath, to a group of breaths, to a time period or to the full recording. An example is the respiratory rate in breaths per minute. In code, a parameter is a `ParameterResult`.

pixel map
  One number for each pixel of an EIT image. An example is the tidal impedance variation of each pixel (`pixel_tiv`). In code, a pixel map is a `PixelMap`.

preset
  A fixed series of processing steps that is part of M3Resp. You run a preset with `session.run_preset`. There are three presets: `"eit"`, `"emg"` and `"multimodal"`.

provenance
  The record of what M3Resp did to the data: the action, the {term}`modality`, the {term}`settings <setting>` and the time (UTC). The {term}`session` keeps this record in `session.provenance`.

quality flag
  The outcome of one quality check on a signal, a breath or a time period. It tells if the check passed and how serious a failure is. It can also give the measured value and the limit. In code, a quality flag is a `QualityFlag`.

R peak
  The highest point of one heartbeat in the {term}`ECG`. M3Resp uses the R peaks to find and remove the ECG from the EMG.

recording
  The data of one {term}`modality` from one file, after M3Resp loads it. Examples are an EIT `.bin` file, or the EMG channels of a Biopac `.txt` file. In code, a recording is an `EITRecording`, an `EMGRecording` or a `VentilatorRecording`.

result
  What a {term}`step` gives back. In a {term}`workflow`, each step keeps its results under a name, and a later step reads them by that name. At the end, the workflow gives back all results in a `WorkflowResult`.

sampling rate
  The number of samples per second of a signal, in Hz. In code, the name is `sample_frequency` or `fs`.

session
  The `M3Session` object of one analysis. It holds the loaded recordings, the signals, the breaths, the parameters, the quality flags and the provenance.

setting
  One value that controls a {term}`step`, for example `low_pass_hz: 500.0`. In a workflow file, the settings of a step are under `with:`. In code, a `StepParameter` describes each setting.

signal
  One series of values over time, with a {term}`sampling rate` and a unit. Examples are the global impedance of an EIT recording and the envelope of one EMG channel. A signal also tells its modality, its channel and its processing state: `"raw"`, `"intermediate"`, `"processed"` or `"derived"`. In code, a signal is a `Signal`.

start time
  The time in seconds at which a {term}`recording` starts on the shared clock. Each recording has one start time in `session.start_times`. When you do not synchronize the recordings, each start time is 0 s.

step
  One operation in a {term}`workflow`, for example `emg.preprocess` or `eit.detect_breaths`. In a workflow file, the step name is under `uses:`. To see all steps, run `m3resp steps`.

synchronization
  The process that puts recordings from different devices on one shared clock. M3Resp gives each recording a {term}`start time` from the {term}`offset` that you enter. All samples of each recording stay.

time window
  A part of a recording between a start time and an end time, in seconds. To slice a recording to a time window, use a step such as `emg.slice_recording`.

ventilator
  The {term}`modality` for airway pressure, flow and volume. The data can come from a ventilator export, or from channels inside an EMG or EIT file.

workflow
  A YAML or JSON file that lists {term}`steps <step>` in the order they run. For each step, it gives the step name (`uses:`), its settings (`with:`) and, if necessary, new names for its results (`out:`). To run a workflow, use `m3resp run <file>`. In code, a loaded workflow is a `WorkflowSpec`.
```
