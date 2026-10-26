"""
Process an EIT recording
========================

In this tutorial, we process one :term:`EIT` recording from the file to the results. We load the file, filter the signal, find the breaths and save the results. At the end, you have the figure below and a folder with result files.

You need M3Resp with the EIT integration (``python -m pip install -e ".[eit]"``).
"""

# %%
# Load the recording
# ------------------
#
# See :doc:`/concepts/session` for what a session holds.
#
# A :term:`session` holds the recordings and all the results of one analysis. We make an empty session and load the EIT file into it. The file is a Dräger ``.bin`` file, so we set ``vendor="draeger"``.

import os
import tempfile

from m3resp import M3Session

# The example recordings are in the tests/data folder of the M3Resp source code.
data_dir = os.environ.get("M3RESP_EXAMPLE_DATA", os.path.join("tests", "data"))

session = M3Session()
session.load_eit(
    os.path.join(data_dir, "m3resp_multimodal_1_eit_draeger.bin"),
    vendor="draeger",
)

# %%
# The session now has the raw EIT data. The global impedance is the sum of all pixels in each EIT image.

global_impedance = session.raw["eit"].global_impedance
print("Sampling rate:", global_impedance.sample_frequency, "Hz")
print("Number of samples:", len(global_impedance.values))

# %%
# The recording is 60 s long, with 20 EIT images per second.
#
# Filter the signal
# -----------------
#
# See :doc:`/concepts/signals` for how M3Resp stores a signal.
#
# :meth:`~m3resp.M3Session.preprocess_eit` finds the respiratory rate and the heart rate. Then it removes the heart signal from the EIT data with an MDN filter (multiple digital notch filter). It also calculates the tidal impedance variation (TIV) and the end-expiratory lung impedance (EELI) of each breath.

session.preprocess_eit()

for signal in session.signals:
    print(f"{signal.name}: {signal.sample_frequency} Hz, unit {signal.unit}")

# %%
# The session now has two :term:`signals <signal>`: the raw global impedance and the filtered global impedance. EIT impedance has no physical unit, so the unit is ``a.u.`` (arbitrary units).
#
# The rates are :term:`parameters <parameter>` of the recording. M3Resp stores rates in Hz. Multiply by 60 to get breaths or beats per minute.

for parameter in session.parameter_results:
    per_minute = 60 * parameter.value
    rate = f"{parameter.value:.3f} Hz = {per_minute:.1f} per minute"
    print(f"{parameter.name}: {rate}")

# %%
# Find the breaths
# ----------------
#
# See :doc:`/concepts/events-and-breaths` for what a breath holds.
#
# :meth:`~m3resp.M3Session.detect_eit_breaths` finds each :term:`breath` in the filtered global impedance. Each breath has a start time, an :term:`extremum` time and an end time, in seconds.

breaths = session.detect_eit_breaths()

print("Number of breaths:", len(breaths))
for breath in breaths[:3]:
    print(
        f"start {breath.start_time:5.2f} s, "
        f"extremum {breath.extremum_time:5.2f} s, "
        f"end {breath.end_time:5.2f} s"
    )

# %%
# The session also stores the breaths, under ``session.events["eit_breaths"]``.
#
# Each breath has a TIV and an EELI value. M3Resp keeps these values per breath in ``session.interval_data``.

tiv = session.interval_data.for_name("continuous_tivs")[0]
first_values = [round(float(value), 1) for value in tiv.values[:3]]
print("TIV of the first three breaths:", first_values)

# %%
# Look at the results
# -------------------
#
# :func:`~m3resp.visualization.plot_eit_processing_summary` shows the filtered global impedance with the breaths, the TIV and EELI of each breath, and the mean TIV of each pixel.
#
# With ``backend="plotly"``, the figure is interactive. Drag over a part of a panel to zoom in, double-click to zoom out, and move the mouse over a line to see its values. ``fig.show()`` shows the figure in a notebook, or in your web browser when you run a Python script. Without ``backend="plotly"``, you get a static matplotlib figure.

from m3resp.visualization import plot_eit_processing_summary

fig = plot_eit_processing_summary(session, backend="plotly")
fig.show()

# %%
# Save the results
# ----------------
#
# See :doc:`/how-to/export-results` for the list of files and their contents.
#
# :meth:`~m3resp.M3Session.export_summary` writes all results of the session to a folder. Here we use a temporary folder. In your own analysis, give the path of your results folder.

output_dir = tempfile.mkdtemp()
session.export_summary(output_dir)

for file_name in sorted(os.listdir(output_dir)):
    print(file_name)

# %%
# Recap
# -----
#
# In this tutorial, we:
#
# 1. loaded an EIT recording into a session with :meth:`~m3resp.M3Session.load_eit`,
# 2. filtered it and calculated the rates, TIV and EELI with :meth:`~m3resp.M3Session.preprocess_eit`,
# 3. found 13 breaths with :meth:`~m3resp.M3Session.detect_eit_breaths`,
# 4. plotted the results and wrote them to files with :meth:`~m3resp.M3Session.export_summary`.
#
# The ``"eit"`` :term:`preset` runs steps 2 and 3 with one call: ``session.run_preset("eit")``.
#
# Next
# ----
#
# - Process an EMG recording in the next tutorial.
# - Change the filter or the results that M3Resp calculates: :doc:`/how-to/eit-processing-settings`.
