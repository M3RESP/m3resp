"""
Combine EIT, EMG and ventilator data
====================================

In this tutorial, we analyse three recordings of the same patient together: EIT, EMG and ventilator data. We put the recordings on one clock, find the breaths in each recording, link the breaths of the three modalities and calculate the time between them. At the end, you have the figure below and a table of linked breaths.

You need M3Resp with the EIT and EMG integrations (``python -m pip install -e ".[all]"``). Do the EIT and EMG tutorials first.
"""

# %%
# Load the recordings
# -------------------
#
# We load the three files into one :term:`session`. The ventilator file is a separate recording with its own clock, so we set ``source="ventilator"``.

import os

from m3resp import M3Session

# The example recordings are in the tests/data folder of the M3Resp source code.
data_dir = os.environ.get("M3RESP_EXAMPLE_DATA", os.path.join("tests", "data"))

session = M3Session()
session.load_eit(
    os.path.join(data_dir, "m3resp_multimodal_1_eit_draeger.bin"),
    vendor="draeger",
)
session.load_emg(os.path.join(data_dir, "m3resp_multimodal_1_emg.Poly5"))
session.load_ventilator(
    os.path.join(data_dir, "m3resp_multimodal_1_ventilator.Poly5"),
    source="ventilator",
)

# %%
# Put the recordings on one clock
# -------------------------------
#
# See :doc:`/concepts/synchronization` for how M3Resp uses start times.
#
# Each recording has its own clock, which starts at 0 s at the start of the file. :meth:`~m3resp.M3Session.synchronize_raw_modalities` gives each recording a :term:`start time` on a shared clock. The start time is in seconds from the start of the reference recording. A negative start time means that the recording started earlier than the reference.
#
# In these files, the EMG recording started 2 s before the EIT and ventilator recordings. So the EMG start time is -2.0 s.

session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": -2.0, "ventilator": 0.0},
    reference_modality="eit",
)
print("Start times (s):", session.start_times)

# %%
# M3Resp does not change or remove samples. Each recording keeps its own clock. M3Resp adds the start time when it compares the breaths of two modalities.
#
# Find the breaths in each recording
# ----------------------------------
#
# We process each :term:`modality` as in the EIT and EMG tutorials. For the ventilator, :meth:`~m3resp.M3Session.detect_ventilator_breaths` finds the breaths in the volume signal.

session.run_preset("eit")
session.run_preset("emg", config={"preprocess": {"channel": 1}})
session.preprocess_ventilator()
session.detect_ventilator_breaths()

for name in ["eit_breaths", "emg_breaths", "ventilator_breaths"]:
    print(f"{name}: {len(session.events[name])} breaths")

# %%
# The figure shows the signals and the breaths of each modality, before and after synchronization. After synchronization, the EMG moves 2 s to the left.

from m3resp.visualization import plot_synchronization_comparison

fig = plot_synchronization_comparison(session)

# %%
# Link the breaths
# ----------------
#
# See :doc:`/concepts/synchronization` for how M3Resp links breaths.
#
# :meth:`~m3resp.M3Session.link_breaths` puts the breaths of the three modalities that belong together into one :term:`linked breath`. M3Resp compares the :term:`extremum` times on the shared clock. Two breaths link when their extremum times differ by 0.5 s or less (``time_tolerance``). A breath with no partner gets a linked breath of its own.

linked_breaths = session.link_breaths(time_tolerance=0.5)

for linked in linked_breaths[:8]:
    parts = ", ".join(
        f"{modality} {breath.extremum_time:.2f} s"
        for modality, breath in linked.breaths.items()
    )
    print(parts)

# %%
# The extremum marks a different moment of the breath in each modality. In the EIT, it is the highest impedance. In the EMG, it is the highest muscle activity. In the ventilator data, it is the highest volume. So the extremum times of one breath are not equal, and some breaths have no partner within 0.5 s.
#
# Calculate the timing between modalities
# ---------------------------------------
#
# See :ref:`breath-timing-parameters` for the definition of each timing parameter.
#
# :meth:`~m3resp.M3Session.compute_breath_timing_parameters` uses the linked breaths. It calculates the delay and the difference in duration between two modalities, for each linked breath. It also calculates the fraction of breaths that both modalities found.

timing_parameters = session.compute_breath_timing_parameters()

for parameter in timing_parameters:
    if parameter.name == "eit_to_emg_delay":
        delay = f"{parameter.value:.2f} {parameter.unit}"
        print(f"Linked breath {parameter.breath_id}: EIT to EMG delay {delay}")

for parameter in timing_parameters:
    if parameter.name.endswith("event_agreement"):
        print(f"{parameter.name}: {parameter.value:.2f}")

# %%
# The delay is the start time of the EMG breath minus the start time of the EIT breath. A positive delay means that the EMG breath comes later. The start time of an EMG breath is the time of the highest muscle activity, as in the EMG tutorial.
#
# Recap
# -----
#
# In this tutorial, we:
#
# 1. loaded EIT, EMG and ventilator recordings into one session,
# 2. gave each recording a start time on a shared clock with :meth:`~m3resp.M3Session.synchronize_raw_modalities`,
# 3. found the breaths in each modality,
# 4. linked the breaths of the three modalities with :meth:`~m3resp.M3Session.link_breaths`,
# 5. calculated the timing parameters between modalities with :meth:`~m3resp.M3Session.compute_breath_timing_parameters`.
#
# :meth:`~m3resp.M3Session.export_summary` writes the linked breaths to ``linked_breaths.csv`` and the timing parameters to ``parameter_results.csv``.
#
# Next
# ----
#
# - Shift the breath times after detection, or skip synchronization: :doc:`/how-to/synchronize-recordings`.
# - Compare other moments of the breath, or other modality pairs: :doc:`/how-to/compare-breath-timing`.
# - Run the same analysis from a workflow file: :doc:`/workflows`.
