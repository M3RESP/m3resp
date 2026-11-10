"""Figure of the breathing-rate and heart-rate detection of EIT images.

Three panels share the rate axis, in breaths or beats per minute:

1. The power spectrum of the summed pixels, divided by its total power, with
   the breathing-rate range (grey) and the breathing rate found (red).
2. The average of the pixel spectra, each divided by its total power.
3. Panel 2 minus panel 1, with the heart-rate range (grey) and the heart
   rate found (red).

The figure reads the in-between results (``captures``) of
``m3resp.eit.processing.detect_rates``.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/plotting/rate_detection.py::
                RateDetectionPlotting.plot
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - A function, `plot_rate_detection`, that reads the rate limits from
      `RateDetectionSettings`.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure, SubFigure

    from m3resp.eit.processing import RateDetectionSettings

MINUTE = 60


def plot_rate_detection(
    settings: RateDetectionSettings,
    *,
    frequencies: np.ndarray,
    normalized_total_power: np.ndarray,
    average_normalized_pixel_power: np.ndarray,
    diff_total_averaged_power: np.ndarray,
    estimated_respiratory_rate: float,
    estimated_heart_rate: float,
    fig: Figure | SubFigure | None = None,
) -> Any:
    """Draw the spectra of the rate detection with both rates.

    Call it with the captures of ``detect_rates``:
    ``plot_rate_detection(settings, **captures)``.

    Args:
        settings (RateDetectionSettings): The settings the rates were found
            with; they give the rate ranges.
        frequencies (np.ndarray): Frequencies of the spectra in Hz.
        normalized_total_power (np.ndarray): Spectrum of the summed pixels,
            divided by its total power.
        average_normalized_pixel_power (np.ndarray): Average of the pixel
            spectra, each divided by its total power.
        diff_total_averaged_power (np.ndarray): The average pixel spectrum
            minus the spectrum of the summed pixels.
        estimated_respiratory_rate (float): The breathing rate found, in Hz.
        estimated_heart_rate (float): The heart rate found, in Hz.
        fig (Figure | SubFigure | None): A matplotlib figure or subfigure to
            draw in. None makes a new figure of 6 x 4.5 inches.

    Returns:
        Any: The matplotlib figure or subfigure. The rate axis is in breaths
            or beats per minute and ends at 1.25 times the highest heart
            rate.

    Raises:
        ImportError: If matplotlib is not installed.
    """

    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    min_respiratory_rate = cast(float, settings.min_respiratory_rate)
    max_respiratory_rate = cast(float, settings.max_respiratory_rate)
    min_heart_rate = cast(float, settings.min_heart_rate)
    max_heart_rate = cast(float, settings.max_heart_rate)

    if fig is None:
        fig = plt.figure(figsize=(6, 4.5))

    axes = fig.subplots(3, 1, sharex=True)
    frequency_range = frequencies < max_heart_rate * 1.25
    rate = frequencies[frequency_range] * MINUTE

    axes[0].set(ymargin=0.2)
    axes[0].plot(
        rate,
        normalized_total_power[frequency_range],
        label="Normalized power",
        color="k",
    )
    axes[1].plot(
        rate,
        average_normalized_pixel_power[frequency_range],
        label="Average normalized power",
        color="k",
    )
    axes[2].plot(
        rate, diff_total_averaged_power[frequency_range], color="k", label="Difference"
    )

    ylim0 = axes[0].get_ylim()
    axes[0].fill_betweenx(
        ylim0,
        min_respiratory_rate * MINUTE,
        max_respiratory_rate * MINUTE,
        color="k",
        alpha=0.2,
        label="RR range",
    )
    axes[0].set(ylim=ylim0)
    axes[0].axvline(
        estimated_respiratory_rate * MINUTE,
        color="r",
        label="Estimated RR",
        zorder=1.9,
        lw=1.5,
    )

    ylim2 = axes[2].get_ylim()
    axes[2].fill_betweenx(
        ylim2,
        min_heart_rate * MINUTE,
        max_heart_rate * MINUTE,
        color="k",
        alpha=0.2,
        label="HR range",
    )
    axes[2].set(ylim=ylim2)
    axes[2].axvline(
        estimated_heart_rate * MINUTE,
        color="r",
        label="Estimated HR",
        zorder=1.9,
        lw=1.5,
    )

    axes[0].set(ylabel="Power (a.u.)", xmargin=0)
    axes[1].set(ylabel="Power (a.u.)")
    axes[1].sharey(axes[0])
    axes[2].set(
        ylim=(-0.2 * ylim2[1], ylim2[1]), ylabel="Diff. (a.u.)", xlabel="Rate (bpm)"
    )

    for axis in axes:
        axis.legend(loc="upper right")
    for axis in fig.axes:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=15))

    # The rates found, in red above panels 1 and 3.
    for axis, found in (
        (axes[0], estimated_respiratory_rate),
        (axes[2], estimated_heart_rate),
    ):
        top = axis.twiny()
        top.set(
            ylim=axis.get_ylim(),
            xlim=axis.get_xlim(),
            xticks=[found * MINUTE],
            xticklabels=[f"{found * MINUTE:.1f} bpm"],
        )
        top.tick_params(axis="x", length=0)
        for label in top.get_xticklabels():
            label.set_color("red")

    fig.suptitle("Rate Detection Results")
    fig.align_ylabels(axes)
    if isinstance(fig, Figure):
        fig.tight_layout()
    return fig
