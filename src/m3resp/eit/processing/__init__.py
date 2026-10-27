"""EIT calculations that m3resp runs itself.

``global_impedance`` sums the pixels of each EIT image into one waveform.
``regional_impedance`` and ``functional_impedance`` sum only the pixels inside
a ``PixelMask``.
"""

from m3resp.eit.processing.impedance import (
    functional_impedance,
    global_impedance,
    regional_impedance,
)

__all__ = ["functional_impedance", "global_impedance", "regional_impedance"]
