"""M3resp-native EMG I/O module."""

from __future__ import annotations

from .loaders import (
    _load_adicht,
    _load_biopac_txt,
    _load_csv,
    _load_mat,
    _load_npy,
    _load_poly5,
    _LoaderMixin,
)
from .vendors.adinstruments import AdichtReader
from .vendors.poly5reader import Poly5Reader

__all__ = [
    "AdichtReader",
    "Poly5Reader",
    "_LoaderMixin",
    "_load_adicht",
    "_load_biopac_txt",
    "_load_csv",
    "_load_mat",
    "_load_npy",
    "_load_poly5",
]
