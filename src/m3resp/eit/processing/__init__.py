"""EIT calculations that m3resp runs itself.

``global_impedance`` sums the pixels of each EIT image into one waveform.
``regional_impedance`` and ``functional_impedance`` sum only the pixels inside
a ``PixelMask``. ``detect_rates`` finds the breathing rate and the heart rate
of EIT images, and ``find_breaths`` finds the breaths of an impedance
waveform. ``mdn_filter`` removes the heart signal from EIT data, and
``mdn_frequency_bands`` gives the bands it removes.
``tidal_impedance_variation`` and ``end_expiratory_lung_impedance`` give the
TIV and the EELI of each breath of an impedance waveform, and
``pixel_tidal_impedance_variation`` and ``pixel_end_expiratory_lung_impedance``
give one map of TIV or EELI values per breath for the pixels of EIT images.
``find_pixel_breaths`` finds the breaths of each pixel from the global
breaths. ``minute_image`` gives the mean pixel TIV over the breaths in a time
window. ``butterworth_filter_images`` filters each pixel of EIT images with a
Butterworth filter. ``breath_intervals_to_breath_events`` and
``breaths_to_interval_data`` change EIT breaths between a list of
``BreathEvent`` objects and an ``IntervalData``. ``preprocess_eit_recording`` runs
these calculations in one call on a loaded recording.
"""

from m3resp.eit.processing.breath_detection import (
    BreathDetectionSettings,
    find_breaths,
)
from m3resp.eit.processing.breath_inputs import (
    breath_intervals_to_breath_events,
    breaths_to_interval_data,
)
from m3resp.eit.processing.butterworth import (
    MAX_EIT_BUTTERWORTH_ORDER,
    butterworth_filter_images,
)
from m3resp.eit.processing.eeli import (
    end_expiratory_lung_impedance,
    pixel_end_expiratory_lung_impedance,
)
from m3resp.eit.processing.impedance import (
    functional_impedance,
    global_impedance,
    regional_impedance,
)
from m3resp.eit.processing.mdn_filter import mdn_filter, mdn_frequency_bands
from m3resp.eit.processing.minute_image import minute_image
from m3resp.eit.processing.pixel_breath import find_pixel_breaths
from m3resp.eit.processing.preprocess import preprocess_eit_recording
from m3resp.eit.processing.rate_detection import (
    RateDetectionSettings,
    detect_rates,
)
from m3resp.eit.processing.tidal_impedance_variation import (
    TIV_METHODS,
    pixel_tidal_impedance_variation,
    tidal_impedance_variation,
)

__all__ = [
    "MAX_EIT_BUTTERWORTH_ORDER",
    "TIV_METHODS",
    "BreathDetectionSettings",
    "RateDetectionSettings",
    "breath_intervals_to_breath_events",
    "breaths_to_interval_data",
    "butterworth_filter_images",
    "detect_rates",
    "end_expiratory_lung_impedance",
    "find_breaths",
    "find_pixel_breaths",
    "functional_impedance",
    "global_impedance",
    "mdn_filter",
    "mdn_frequency_bands",
    "minute_image",
    "pixel_end_expiratory_lung_impedance",
    "pixel_tidal_impedance_variation",
    "preprocess_eit_recording",
    "regional_impedance",
    "tidal_impedance_variation",
]
