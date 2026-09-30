from .io import _LoaderMixin
from .processing import (
    _BaselineMixin,
    _CoreMixin,
    _DefaultsMixin,
    _EcgMixin,
    _QualityMixin,
)


class ReSurfEMG(
    _CoreMixin, _EcgMixin, _BaselineMixin, _QualityMixin, _LoaderMixin, _DefaultsMixin
):
    """Ported methods from the `ReSurfEMG` package."""


__all__ = [
    "ReSurfEMG",
]
