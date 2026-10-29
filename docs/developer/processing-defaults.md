# Processing defaults

This page lists the default settings of the EIT, EMG and ventilator processing in one place. It also lists the settings where two places use different values for the same thing.

M3Resp gets most of its processing from `eitprocessing` (EIT) and `resurfemg` (EMG and part of the ventilator processing). Each package has its own defaults. M3Resp shows some of them as step settings. Others are fixed in the code of the package.

## Checked versions

The values come from the code on 2026-10-01. Both packages come from the `m3resp-integration` branch of the M3RESP copies, at the commits in `uv.lock`:

- eitprocessing 1.8.7, commit `6d6da6a825a08201ad01c0a7b68f4de07d4501f3`
- ReSurfEMG 1.1.1.dev47, commit `c63668689030e4581d5f985e7d09d3a8c01e7a77`

Check the values again after an update of either package.

## The "Set by" column

| Value | Meaning |
|---|---|
| m3resp | A step setting. It shows in a workflow file and in the GUI. |
| m3resp (undeclared) | M3Resp gives it to the package, but the step does not list it as a setting. You can set it in a workflow file, but the GUI does not show it. |
| m3resp (fixed) | A value in the M3Resp code. You cannot change it. |
| eitprocessing, ReSurfEMG | A value in the code of the package. You cannot change it from a workflow file. |

## 1. Open points

Each point is one kind of value with different values in two places. The last column is a proposal for the team.

| # | What | Current values | Proposal |
|---|---|---|---|
| D1 | Butterworth filter order | EIT `eit.butterworth_filter`: 4 (m3resp's own choice; eitprocessing has no default). EMG band-pass: 3 (ReSurfEMG, cannot be changed). | One default order for all band filters, and make the EMG order a setting. |
| D2 | Shortest breath accepted | EIT `eit.detect_breaths` and `eit.pixel_breaths`: 0.67 s. EMG `emg.detect_breaths`: 0.5 s (undeclared). ReSurfEMG's own EMG class: 0.2 s. Ventilator `ventilator.detect_breaths` and `ventilator.normalize_breaths`: 0.5 s. Occlusions in `ventilator.find_occluded_breaths`: 0.1 s (m3resp, fixed). | One value in seconds for all three modalities, as a setting of each breath detection step. |
| D3 | Highest breathing rate allowed | EIT rate search, adult: 60 /min (neonate 85 /min). Ventilator `ventilator.detect_pressure_breaths`: shortest gap 2 s, so 30 /min. Occlusions in `ventilator.find_occluded_breaths`: shortest gap 0.5 s (m3resp, fixed). A 0.67 s shortest breath (D2) allows 90 /min. | Take the limits from `subject_type` (adult/neonate), as EIT already does. |
| D4 | Highest heart rate allowed | EIT rate search, adult: 200 /min (neonate 210 /min). EMG `emg.ecg_detect_peaks`: shortest gap between beats is a third of a second, so 180 /min. It can be changed with `peak_distance_seconds`. | Same heart-rate range for both devices, taken from `subject_type`. |
| D5 | ECG gate width, when not set | `emg.ecg_gating` takes `gate_width_seconds` or `gate_width_samples`. When neither is set it uses 205 samples (ReSurfEMG function default). That is 0.1 s only at 2048 Hz; at 4000 Hz it is 0.05 s. ReSurfEMG's own EMG class falls back to 0.1 s at any sampling rate. | Fall back to 0.1 s instead of 205 samples. |
| D6 | How a gate is filled | `emg.ecg_gating`: 1 (straight line between the gate edges; ReSurfEMG function default). ReSurfEMG's own EMG class and ECG-removal workflow: 3 (running RMS). | Choose one method. ReSurfEMG uses method 3 in its own workflow. |
| D7 | Slope window for breath onset/offset | `emg.onoffpeak_slope_extrapolation`: 0.5 s. ReSurfEMG's peak-set class: 0.2 s. | Choose one value. ReSurfEMG uses 0.2 s. |
| D8 | Same step, different default from Python | `ventilator.detect_breaths`: the step setting says 0.5 s, but calling the Python function directly uses 0.25 s, so the same data gives different breath counts. `emg.ecg_detect_peaks`: the step setting is empty (ReSurfEMG then uses a third of a second), while the Python function says 0.33 s. Same result today, but written in two places. | Make the Python default and the step setting the same value. |

All EMG and ventilator baselines use a window of 7.5 s, a step of 0.2 s and the 33rd percentile, as in ReSurfEMG: `emg.moving_baseline`, `emg.slopesum_baseline`, the `"emg"` preset, `M3Session.postprocess_emg` without a baseline, and `ventilator.pocc_intervals`. The EIT breath detection uses a moving average of 15 s.

## 2. Values that M3Resp chose

The packages give no default for these settings, so M3Resp chose the values. Each value needs a reason or a reference in the docs.

| Step | Setting | Value |
|---|---|---|
| `eit.butterworth_filter` | `mode` / `lowpass_hz` / `highpass_hz` / `order` | `lowpass` / 1.0 Hz / 0.05 Hz / 4 (1 to 10) |
| `emg.detect_breaths` | shortest breath (`min_breath_width_seconds`) | 0.5 s |
| `ventilator.detect_breaths` | `breath_width_seconds` | 0.5 s |
| `ventilator.detect_pressure_breaths` | `smoothing_seconds` / `min_depth` / `min_interval_seconds` | 0.2 s / 0.15 / 2.0 s |
| `ventilator.find_occluded_breaths` | `peep` | median airway pressure |
| `emg.envelope` in the `"emg"` preset | envelope window and method after ECG gating | the ones recorded by `session.preprocess_emg` (0.25 s / RMS unless set) |

## 3. All defaults, by modality

### EIT (eitprocessing)

| Step | Setting | Value | Set by |
|---|---|---|---|
| `eit.detect_breaths` | shortest breath | 0.67 s | m3resp (same as eitprocessing) |
| | moving-average window | 15 s, Blackman shape | m3resp (same as eitprocessing) |
| | smallest breath, as a fraction of the typical breath | 0.25 | m3resp (same as eitprocessing) |
| | removal of bad data: window / percentile / factor | 0.5 s / 5th / ×4 | m3resp (same as eitprocessing) |
| `eit.detect_rates` | `subject_type` | adult | m3resp |
| | Welch window | 30 s (shortest allowed 10 s) | m3resp (same as eitprocessing), can be set with `welch_window_seconds` |
| | Welch overlap | 0.5 | m3resp (same as eitprocessing) |
| | breathing rate, adult / neonate | 6 to 60 /min / 15 to 85 /min | m3resp (same as eitprocessing) |
| | heart rate, adult / neonate | 40 to 200 /min / 90 to 210 /min | m3resp (same as eitprocessing) |
| `eit.mdn_filter` | upper noise limit | 220 /min | m3resp (same as eitprocessing), can be set with `noise_frequency_limit_hz` |
| | notch distance | 10 /min | m3resp (same as eitprocessing), can be set with `notch_distance_hz` |
| | filter order | 10 | m3resp (same as eitprocessing), can be set with `order` (1 to 10) |
| `eit.butterworth_filter` | see section 2 | | m3resp |
| `eit.continuous_tiv` | `tiv_method` | inspiratory | m3resp (same as eitprocessing) |
| `eit.pixel_breaths` | `phase_correction_mode` | negative amplitude | m3resp (same as eitprocessing) |
| | `minimum_duration_seconds` | 0.67 s | m3resp (same as eitprocessing) |
| `eit.pixel_tiv` | `tiv_timing` | continuous | m3resp (eitprocessing: pixel) |
| | `tiv_method` | inspiratory | m3resp (same as eitprocessing) |
| `eit.pixel_eeli` | `eeli_timing` | continuous | m3resp (no eitprocessing version) |
| `eit.roi_tiv_lungspace`, `eit.roi_amplitude_lungspace` | `threshold` | 0.15 | m3resp (same as eitprocessing) |
| `eit.roi_watershed` | `threshold_fraction` | 0.15 | m3resp (same as eitprocessing) |
| `eit.roi_filter_by_size` | `min_region_size` / `connectivity` | 10 pixels / 1 (4 neighbours) | m3resp (same as eitprocessing) |

### EMG (ReSurfEMG)

| Step | Setting | Value | Set by |
|---|---|---|---|
| `emg.preprocess`, `emg.select_channel` | channel | picked from the channel labels | m3resp |
| `emg.preprocess`, `emg.bandpass_filter` | band-pass edges | 20 Hz to the lower of 500 Hz and 0.95 × half the sampling rate | m3resp, same as ReSurfEMG |
| | band-pass order | 3 | ReSurfEMG |
| `emg.preprocess`, `emg.notch_filter` | harmonic notch (EIT frame-rate comb) | off in `emg.preprocess` unless a base frequency is given | m3resp |
| | notch quality factor / upper limit / applied | 30 / half the sampling rate / after the band-pass | m3resp |
| `emg.envelope` | envelope window / method | 0.25 s / RMS | m3resp, same as ReSurfEMG |
| `emg.detect_breaths` | shortest breath | 0.5 s | m3resp (undeclared) |
| | prominence: factor × (75th + 50th percentile of the envelope above baseline) | 0.5 | m3resp (undeclared), same as ReSurfEMG |
| | merge peaks closer than the shortest breath | no | m3resp (undeclared) |
| `emg.moving_baseline` | window / step / percentile | 7.5 s / 0.2 s / 33 | m3resp |
| `emg.slopesum_baseline` | window / step / percentile / extra percentile | 7.5 s / 0.2 s / 33 / 25 | m3resp |
| | smoothing window / percentile window | 0.5 s / 1.0 s | m3resp |
| `emg.ecg_detect_peaks` | `peak_fraction` | 0.4 | m3resp (same as ReSurfEMG) |
| | band-pass before detection | 1 to 500 Hz | ReSurfEMG |
| | shortest peak width / shortest gap between peaks | 1 ms / 0.33 s | ReSurfEMG, can be set |
| `emg.ecg_gating` | gate width / fill method | 205 samples / 1 | m3resp (same as the ReSurfEMG function) |
| `emg.ecg_wavelet_denoising` | levels / wavelet / threshold / hard thresholding | 4 / db2 / 4.5 / yes | m3resp (same as ReSurfEMG) |
| `emg.onoffpeak_slope_extrapolation` | `slope_window_seconds` | 0.5 s | m3resp |
| `emg.area_under_baseline` | `window_seconds` | 5 s | m3resp (same as ReSurfEMG) |
| `emg.percentage_under_baseline` | window / limit | the breath's own onset to offset / 40 % | m3resp (limit same as ReSurfEMG) |
| `emg.detect_local_high_aub` | percentile / factor | 75 / ×4 | m3resp (same as ReSurfEMG) |
| `emg.detect_extreme_time_products` | upper percentile and factor / lower percentile and factor | 95, ×10 / 5, ×0.1 | m3resp (same as ReSurfEMG) |
| `emg.evaluate_respiratory_rates` | `minimum_fraction` | 0.1 | m3resp (same as ReSurfEMG) |
| `emg.interpeak_dist` | `threshold` | 1.1 | m3resp (same as ReSurfEMG) |
| `emg.evaluate_bell_curve_error` | largest bell-curve error for a valid breath | 40 % | m3resp (fixed), same as ReSurfEMG |
| `emg.evaluate_event_timing` | how far the EMG peak may come before the ventilator peak: least / most | −0.5 s / 2.0 s | m3resp (fixed) |
| `emg.snr_pseudo` | `minimum_snr` | none (measure only) | m3resp |
| respiratory rate | outlier percentile / factor | 33 / ×3 | m3resp (fixed), same as ReSurfEMG |

### Ventilator

| Step | Setting | Value | Set by |
|---|---|---|---|
| `ventilator.detect_breaths` | `breath_width_seconds` | 0.5 s (0.25 s from Python, see D8) | m3resp |
| | height / prominence, first pass (× 90th percentile of volume) | 0.25 / 0.10 | m3resp (fixed) |
| | height / prominence, second pass (× 90th percentile of volume) | 0.5 / 0.5 | m3resp (fixed) |
| `ventilator.normalize_breaths` | `breath_width_seconds` | 0.5 s | m3resp |
| `ventilator.detect_pressure_breaths` | see section 2 | | m3resp |
| `ventilator.find_occluded_breaths` | prominence: factor × (PEEP − lowest pressure) | 0.8 | m3resp (fixed), same as ReSurfEMG |
| | shortest occlusion / shortest gap between occlusions | 0.1 s / 0.5 s | m3resp (fixed) |
| `ventilator.pocc_intervals` | window / step / percentile | 7.5 s / 0.2 s / 33 | m3resp |
| `ventilator.pocc_quality` | dP thresholds at 10 % / 90 % / 90 % scaled | 0.0 / 2.0 / 0.8 | m3resp (same as ReSurfEMG) |
| `ventilator.pocc_time_product` | add area under baseline / its window | yes / 5 s | m3resp |

## 4. Other findings

- `emg.preprocess` and `emg.detect_breaths` list only `variant` as a setting. You can set their filter limits, envelope window, channel and shortest breath in a workflow file, but the GUI does not show them.
- Some default values are numbers of samples, for example the gate width in D5. So their duration in seconds depends on the sampling rate.

Continue with {doc}`migration`.
