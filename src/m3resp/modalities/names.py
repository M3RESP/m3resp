"""The names of the modalities, and the spellings accepted for them.

Kept free of other m3resp imports, so any module can use it without pulling
in the rest of the package.
"""

from __future__ import annotations

#: Canonical name for the ventilator modality. ``"vent"`` was the Stage 1
#: internal spelling while ``"ventilator"`` was used in the docs, by
#: ``M3Session.link_breaths``, and by ``Signal.modality`` - so a breath could
#: carry ``modality="vent"`` while the ``LinkedBreath`` holding it was keyed
#: ``"ventilator"``. Everything now canonicalizes here.
VENTILATOR = "ventilator"

#: Accepted spellings that normalize to :data:`VENTILATOR`. ``"vent"`` is kept
#: indefinitely: it shipped as a ``session.raw`` key and as a pipeline-spec
#: parameter value, so specs and user code still pass it.
_VENTILATOR_ALIASES = frozenset({"vent", "ventilator", "ventilation"})


def normalize_modality(modality: str) -> str:
    """Canonicalize a modality name, mapping every ventilator spelling to
    :data:`VENTILATOR`."""

    normalized = str(modality).lower()
    if normalized in _VENTILATOR_ALIASES:
        return VENTILATOR
    return normalized
