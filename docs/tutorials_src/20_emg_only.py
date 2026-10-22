"""
Process an EMG recording
========================

In this tutorial, we process one :term:`EMG` recording of the respiratory muscles. We load the file, remove the heart signal, find the breaths and look at the results per breath. At the end, you have the figure below.

You need M3Resp with the EMG integration (``python -m pip install -e ".[emg]"``).
"""

# %%
# Load the recording
# ------------------
#
# See :doc:`/concepts/session` for what a session holds.
#
# We make an empty :term:`session` and load the EMG file into it. The file is a ``.Poly5`` file.

import os

from m3resp import M3Session

# The example recordings are in the tests/data folder of the M3Resp source code.
data_dir = os.environ.get("M3RESP_EXAMPLE_DATA", os.path.join("tests", "data"))

session = M3Session()
emg_file = os.path.join(data_dir, "m3resp_multimodal_1_emg.Poly5")
recording = session.load_emg(emg_file)

metadata = recording["metadata"]
print("Sampling rate:", metadata["fs"], "Hz")
print("Channels:", metadata["labels"])
print("Units:", metadata["units"])

# %%
# The file has two :term:`channels <channel>`. Channel 1 has the EMG of the diaphragm, so we use channel 1. The channel numbers start at 0.
#
# Process the EMG
# ---------------
#
# See :doc:`/concepts/signals` for how M3Resp stores a signal.
#
# The ``"emg"`` :term:`preset` runs all processing steps in the correct order:
#
# 1. A band-pass filter from 20 Hz to 500 Hz.
# 2. Detection of the :term:`R peaks <R peak>` of the :term:`ECG` in the EMG channel.
# 3. ECG gating: M3Resp removes a short part of the signal at each R peak and fills the gap.
# 4. The RMS :term:`envelope` of the gated signal, with a window of 0.25 s.
# 5. The :term:`baseline` of the envelope, with a window of 7.5 s that moves in steps of 0.2 s.
# 6. Detection of the breaths in the envelope, above the baseline.
# 7. The calculation of the parameters per breath and the quality checks.

session.run_preset("emg", config={"preprocess": {"channel": 1}})

for signal in session.signals:
    print(f"{signal.name}: {signal.sample_frequency} Hz")

# %%
# The session now has four :term:`signals <signal>` of channel 1: the raw signal, the band-pass filtered signal, the envelope and the signal after ECG gating.
#
# Look at the breaths
# -------------------
#
# See :doc:`/concepts/events-and-breaths` for what a breath holds.
#
# The preset stores the breaths in ``session.events["emg_breaths"]`` and the R peaks of the ECG in ``session.events["ecg_peaks"]``.

breaths = session.events["emg_breaths"]
print("Number of breaths:", len(breaths))
print("Number of R peaks:", len(session.events["ecg_peaks"]))
extremum_times = [round(breath.extremum_time, 2) for breath in breaths[:5]]
print("Extremum times of the first five breaths (s):", extremum_times)

# %%
# Each EMG breath is at the highest point of the envelope (the :term:`extremum`). The default EMG detection finds only this point, so the start time and the end time of each EMG breath are equal to the extremum time.
#
# The figure shows each step of the processing. The red lines mark the breaths.

from m3resp.visualization import plot_session_overview

fig = plot_session_overview(session, backend="plotly")
fig.show()

# %%
# Look at the parameters
# ----------------------
#
# See :doc:`/concepts/parameters` for what a parameter holds.
#
# The last step of the preset calculates the :term:`parameters <parameter>`. Some parameters have one value for the recording. Other parameters have one value per breath.

rate = session.parameter_results.for_name("respiratory_rate")[0]
print(f"Respiratory rate: {rate.value:.1f} {rate.unit}")

amplitude = session.parameter_results.for_name("amplitude")[0]
first_values = [round(float(value), 2) for value in amplitude.value[:3]]
print("Amplitude of the first three breaths:", first_values)

# %%
# Look at the quality checks
# --------------------------
#
# See :doc:`/concepts/quality` for what a quality flag holds.
#
# The preset also runs quality checks. Each check gives a :term:`quality flag` that passed or failed.

for flag in session.quality:
    print(f"{'passed' if flag.passed else 'failed'}: {flag.name}")

# %%
# Some checks need ventilator data. This recording has no ventilator data, so these checks fail. The ``message`` of each flag gives the reason.

for flag in session.quality:
    if flag.message:
        print(f"{flag.name}: {flag.message}")

# %%
# Recap
# -----
#
# In this tutorial, we:
#
# 1. loaded an EMG recording with :meth:`~m3resp.M3Session.load_emg`,
# 2. ran the ``"emg"`` preset on channel 1 with :meth:`~m3resp.M3Session.run_preset`: filter, ECG removal, envelope, baseline, breath detection and parameters,
# 3. looked at the breaths, the parameters and the quality flags.
#
# To save the results to files, use :meth:`~m3resp.M3Session.export_summary`, as in the EIT tutorial.
#
# Next
# ----
#
# - Put EMG, EIT and ventilator recordings on one clock: :doc:`/how-to/synchronize-recordings`.
# - Change the filter, the envelope or the ECG removal: :doc:`/how-to/emg-processing-settings`.
