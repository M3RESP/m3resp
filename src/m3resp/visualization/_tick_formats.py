"""Number formats of the colour bar that leave out the sign.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/plotting/helpers.py::AbsolutePercentFormatter;
                AbsoluteScalarFormatter
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.ticker import PercentFormatter, ScalarFormatter


class AbsolutePercentFormatter(PercentFormatter):
    """Write numbers as percentages without their sign: -0.5 is "50%"."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Return the text of one tick."""
        return super().__call__(abs(x), pos)


class AbsoluteScalarFormatter(ScalarFormatter):
    """Write numbers without their sign: -2 is "2"."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Return the text of one tick."""
        return super().__call__(abs(x), pos)
