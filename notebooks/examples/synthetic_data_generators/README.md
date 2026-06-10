# Unified Synthetic Data Generators

This folder contains one maintained synthetic data generator for M3Resp examples:

- `unified_generator.py`: reusable example module for generating EIT, EMG, and
  ventilator synthetic recordings.
- `synthetic_generator_config.yaml`: editable YAML configuration for the
  generator.
- `unified_synthetic_data_generation.ipynb`: notebook showing the same workflow
  interactively.

The old split EIT and EMG example generators were removed. New examples should
use this unified generator.

## Features

The generator can create a complete synthetic respiratory dataset with any
combination of:

- **EIT data**
  - realistic breathing waveform with breath-to-breath variability;
  - optional cardiac oscillation and measurement noise;
  - optional named baseline drift component;
  - 32 x 32 synthetic lung-shaped pixel impedance frames;
  - Draeger-compatible `.bin` export;
  - portable `.npy`, `.csv`, and component `.npz` exports.

- **EMG data**
  - generated through `resurfemg.pipelines.synthetic_data.simulate_raw_emg`;
  - configurable sampling rate, channel amplitudes, noise, drift, heart rate,
    ECG acceleration, and muscle timing constants;
  - portable `.npy` and `.csv` exports;
  - defensive handling for known ReSurfEMG ECG/EMG length mismatch behavior.

- **Ventilator data**
  - generated through `resurfemg.pipelines.synthetic_data.simulate_ventilator_data`;
  - configurable sampling rate, driving pressure, muscle pressure amplitude, and
    respiratory pattern;
  - pressure, flow, and volume outputs;
  - portable `.npy` and `.csv` exports plus `p_mus` `.npy`.

The returned `SyntheticDataset` includes `eit`, `emg`, and `ventilator` records
when enabled. Each record contains:

- `time`: time vector in seconds;
- `array`: generated samples;
- `sample_frequency`: sampling rate in Hz;
- `labels`: channel or data labels;
- `units`: units for each label;
- `paths`: written output file paths;
- `metadata`: modality-specific metadata.

## Dependencies

EIT-only generation uses the base project dependencies:

```bash
pip install -e ".[dev]"
```

EMG and ventilator generation require ReSurfEMG:

```bash
pip install -e ".[emg]"
```

or, for all optional integrations:

```bash
pip install -e ".[all,dev]"
```

If ReSurfEMG is not installed and EMG or ventilator generation is enabled, the
generator raises a clear runtime error.

## Run From The Notebook

Open:

```text
notebooks/examples/synthetic_data_generators/unified_synthetic_data_generation.ipynb
```

Run the cells from top to bottom.

The notebook first locates this folder, then loads:

```python
config_path = os.path.join(generator_dir, "synthetic_generator_config.yaml")
config = load_synthetic_generator_config(config_path)
dataset = generate_synthetic_dataset(config)
```

The YAML file is the source of truth for input values. Edit its modality toggles
to choose EIT, EMG, ventilator, or any combination.

## Run From Python

From this folder:

```bash
python unified_generator.py
```

This loads `synthetic_generator_config.yaml` from this folder and writes outputs
using the values in that file.

You can also pass a different YAML file:

```bash
python unified_generator.py path/to/custom_config.yaml
```

For the same YAML-driven run from Python:

```python
import os

from unified_generator import (
    generate_synthetic_dataset,
    load_synthetic_generator_config,
)

config = load_synthetic_generator_config(
    os.path.join(os.getcwd(), "synthetic_generator_config.yaml")
)
dataset = generate_synthetic_dataset(config)

print(dataset.eit.paths)
```

## YAML Configuration

Edit `synthetic_generator_config.yaml` to control the generated data.

Top-level keys:

- `duration_seconds`: total recording length.
- `seed`: deterministic random seed.
- `output_dir`: output directory. Relative paths are resolved relative to the
  YAML file location.
- `basename`: prefix used for generated files.
- `timestamp_output_dir`: when `true`, create one run folder inside
  `output_dir` named with date and time to seconds.
- `generate_eit`: enable EIT generation.
- `generate_emg`: enable EMG generation.
- `generate_ventilator`: enable ventilator generation.
- `write_native_outputs`: write native Poly5 EMG/Vent outputs in addition to
  portable `.npy` and `.csv` files.

Respiratory pattern:

```yaml
respiratory:
  respiratory_rate_bpm: 14.0
  respiratory_rate_variation_bpm: 2.0
  ie_ratio: 0.5
  occlusion_times_seconds:
    - 45.0
```

Drift:

```yaml
drift:
  enabled: true
  amplitude: 0.18
  kind: sinusoidal
```

Supported drift kinds are:

- `sinusoidal`
- `linear`
- `constant`
- `time_shift`

To model drift as a simple shift of the generated EIT signal in time, use:

```yaml
drift:
  enabled: true
  kind: time_shift
  time_shift_seconds: 0.5
  time_shift_fill_mode: edge
```

Positive `time_shift_seconds` delays the signal. Negative values advance it.
`time_shift_fill_mode` controls samples exposed at the beginning or end of the
shifted signal and can be `edge` or `zero`. For this mode, the exported `drift`
component is the difference between the shifted and unshifted signal so that the
component sum remains equal to the generated EIT signal.

EIT settings:

```yaml
eit:
  sample_frequency_hz: 20.0
  format_name: original
  base_impedance_au: 10.0
  tidal_amplitude_au: 0.9
  cardiac_amplitude_au: 0.04
  heart_rate_bpm: 72.0
  noise_std_au: 0.02
```

EMG settings:

```yaml
emg:
  sample_frequency_hz: 2048
  channel_amplitudes_uv:
    - 0.2
    - 5.0
  drift_amplitude_uv: 100.0
  noise_amplitude_uv: 2.0
  heart_rate_bpm: 72.0
  ecg_acceleration: 1.6
```

Ventilator settings:

```yaml
ventilator:
  sample_frequency_hz: 100
  driving_pressure_cm_h2o: 8.0
  muscle_pressure_amplitude_cm_h2o: 5.0
```

## Timestamped Run Folders

By default, generated files are written into a new timestamped folder under
`output_dir`.

For example:

```yaml
output_dir: ../../../data/source/synthetic_demo
timestamp_output_dir: true
```

creates output like:

```text
data/source/synthetic_demo/20260609_153045/
```

The timestamp format is:

```text
YYYYMMDD_HHMMSS
```

If two runs start in the same second, a suffix is added:

```text
YYYYMMDD_HHMMSS_01
```

The actual run folder path is available after generation:

```python
dataset.provenance["output_dir"]
```

Set this to `false` when exact output paths are needed for tests or scripted
overwrites:

```yaml
timestamp_output_dir: false
```

## Output Files

For `basename: m3resp_demo`, EIT generation writes these files inside the run
folder:

- `m3resp_demo_eit_pixels.npy`: EIT frames with shape
  `(n_samples, 32, 32)`;
- `m3resp_demo_eit_global.csv`: summed global impedance time series;
- `m3resp_demo_eit_components.npz`: named signal components including
  `baseline`, `breathing`, `drift`, `cardiac`, and `noise`;
- `m3resp_demo_eit_draeger.bin`: Draeger-compatible binary file;
- `m3resp_demo_metadata.json`: provenance and output metadata.

EMG generation writes:

- `m3resp_demo_emg.npy`;
- `m3resp_demo_emg.csv`.

Ventilator generation writes:

- `m3resp_demo_ventilator.npy`;
- `m3resp_demo_ventilator.csv`;
- `m3resp_demo_ventilator_p_mus.npy`.

## Native EMG And Ventilator Output

Portable `.npy` and `.csv` files are always written for EMG and ventilator data.

Native EMG/Vent output writes Poly5 files with channel labels, units, sample
rate, and float32 sample data. If an installed ReSurfEMG version exposes a
native `write_synthetic_recording` helper, the generator uses it. Otherwise,
the example generator writes ReSurfEMG-readable Poly5 files directly.

To write Poly5 alongside portable exports, use:

```yaml
write_native_outputs: true
```

## Troubleshooting

If EMG or ventilator generation says ReSurfEMG is missing, install:

```bash
pip install -e ".[emg]"
```

If a ReSurfEMG version produces ECG/EMG length mismatch errors internally, this
generator retries EMG generation with `ecg_acceleration=1.0` and normalizes the
result to the expected sample count.

If a YAML key is misspelled, loading fails with an explicit unknown-key error.
