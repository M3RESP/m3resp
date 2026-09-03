"""Structural types describing what the core mixin calls on the defaults mixin.

Mirrors `m3resp.adapters.resurfemg_adapter._protocols`: the mixins are split
for navigability, so each one needs a `self` annotation naming the methods it
relies on from its siblings.
"""

from __future__ import annotations

from typing import Any, Protocol


class _DefaultsProtocol(Protocol):  # noqa: PYI046 -- used via `cast()` in _core.py
    """What `_CoreMixin` calls on `_DefaultsMixin`."""

    def _preprocess_default(self, recording: Any, **kwargs: Any) -> Any: ...

    def _detect_breaths_default(
        self, processed_ventilator: Any, **kwargs: Any
    ) -> Any: ...
