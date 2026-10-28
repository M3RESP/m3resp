"""EIT calculations that m3resp runs itself.

``global_impedance`` sums the pixels of each EIT image into one waveform.
``regional_impedance`` and ``functional_impedance`` sum only the pixels inside
a ``PixelMask``. ``detect_rates`` finds the breathing rate and the heart rate
of EIT images, and ``find_breaths`` finds the breaths of an impedance
waveform. ``mdn_filter`` removes the heart signal from EIT data, and
``mdn_frequency_bands`` gives the bands it removes.
"""

from m3resp.eit.processing.breath_detection import (
    BreathDetectionSettings,
    find_breaths,
)
from m3resp.eit.processing.impedance import (
    functional_impedance,
    global_impedance,
    regional_impedance,
)
from m3resp.eit.processing.mdn_filter import mdn_filter, mdn_frequency_bands
from m3resp.eit.processing.rate_detection import (
    RateDetectionSettings,
    detect_rates,
)

__all__ = [
    "BreathDetectionSettings",
    "RateDetectionSettings",
    "detect_rates",
    "find_breaths",
    "functional_impedance",
    "global_impedance",
    "mdn_filter",
    "mdn_frequency_bands",
    "regional_impedance",
]
