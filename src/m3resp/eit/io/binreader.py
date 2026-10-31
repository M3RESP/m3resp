"""Read numbers and text from a binary EIT file.

Draeger ``.bin`` and Sentec ``.zri`` files store each value as a fixed number
of bytes: for example 4 bytes for a 32-bit decimal number. :class:`BinReader`
reads the next value (or the next block of values) from an open file and
moves on to the value after it.

The byte order tells which byte of a value comes first. Sentec files use
``"little"``. With ``endian=None`` the reader uses the byte order of the
computer that runs the code.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/loading/binreader.py::BinReader
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Docstrings are reworded.
    - `cast` of `read_single` and `read_list` has the type hint
      `Callable[[Any], T]`, so type checkers accept `cast(value)`.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import struct
from collections.abc import Callable
from dataclasses import dataclass
from mmap import mmap
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
N = TypeVar("N", bound=np.number)


@dataclass
class BinReader:
    """Read values one after another from an open binary file.

    Args:
        file_handle (io.BufferedReader | mmap): The open file, for example
            the result of ``open(path, "rb")``.
        endian (Literal["little", "big"] | None): The byte order of the
            values. ``None`` uses the byte order of this computer.
    """

    file_handle: io.BufferedReader | mmap
    endian: Literal["little", "big"] | None = None

    def read_single(self, type_code: str, cast: Callable[[Any], T]) -> T:
        """Read one value.

        The type code is one letter of the Python ``struct`` module, for
        example ``"i"`` for a 32-bit whole number or ``"f"`` for a 32-bit
        decimal number. See
        https://docs.python.org/3/library/struct.html#format-characters.

        Args:
            type_code (str): The ``struct`` letter of the value.
            cast (Callable[[Any], T]): The Python type of the result, for
                example ``int`` or ``float``.

        Returns:
            T: The value.

        Raises:
            ValueError: If ``endian`` is not ``"little"``, ``"big"`` or
                ``None``.
        """
        data = self._read_full_type_code(type_code)
        return cast(data[0])

    def read_list(
        self, type_code: str, cast: Callable[[Any], T], length: int
    ) -> list[T]:
        """Read several values of the same type.

        Args:
            type_code (str): The ``struct`` letter of the values.
            cast (Callable[[Any], T]): The Python type of each value.
            length (int): The number of values.

        Returns:
            list[T]: The values, in file order.

        Raises:
            ValueError: If ``endian`` is not ``"little"``, ``"big"`` or
                ``None``.
        """
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return [cast(d) for d in data]

    def read_array(
        self,
        type_code: str,
        cast: type[N],
        length: int,
    ) -> NDArray[N]:
        """Read several values of the same type into a NumPy array.

        Args:
            type_code (str): The ``struct`` letter of the values.
            cast (type[N]): The NumPy type of the array, for example
                ``np.float32``.
            length (int): The number of values.

        Returns:
            NDArray[N]: The values, in file order.

        Raises:
            ValueError: If ``endian`` is not ``"little"``, ``"big"`` or
                ``None``.
        """
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return np.array(data, dtype=cast)

    def read_string(self, length: int = 1) -> str:
        """Read text of a fixed number of characters.

        Spaces and other white space at the end of the text are removed.

        Args:
            length (int): The number of characters (bytes) to read.

        Returns:
            str: The text.

        Raises:
            ValueError: If ``endian`` is not ``"little"``, ``"big"`` or
                ``None``.
        """
        full_type_code = f"{length}s"
        data = self._read_full_type_code(full_type_code)
        return data[0].decode().rstrip()

    string = read_string

    def _read_full_type_code(self, full_type_code: str) -> tuple[Any, ...]:
        """Read the bytes for ``full_type_code`` and unpack them."""
        if self.endian:
            if self.endian not in ["little", "big"]:
                msg = (
                    f"Endian type '{self.endian}' not recognized. "
                    "Allowed values are 'little' and 'big'."
                )
                raise ValueError(msg)

            prefix = "<" if self.endian == "little" else ">"
            full_type_code = prefix + full_type_code

        data_size = struct.calcsize(full_type_code)
        packed_data = self.file_handle.read(data_size)
        return struct.unpack(full_type_code, packed_data)

    def float32(self) -> float:
        """Read one 32-bit decimal number."""
        return self.read_single(type_code="f", cast=float)

    def float64(self) -> float:
        """Read one 64-bit decimal number."""
        return self.read_single(type_code="d", cast=float)

    def npfloat32(self, length: int = 1) -> NDArray[np.float32]:
        """Read ``length`` 32-bit decimal numbers into an array."""
        return self.read_array(type_code="f", cast=np.float32, length=length)

    def npfloat64(self, length: int = 1) -> NDArray[np.float64]:
        """Read ``length`` 64-bit decimal numbers into an array."""
        return self.read_array(type_code="d", cast=np.float64, length=length)

    def int32(self) -> int:
        """Read one 32-bit whole number (can be negative)."""
        return self.read_single(type_code="i", cast=int)

    def npint32(self, length: int = 1) -> NDArray[np.int32]:
        """Read ``length`` 32-bit whole numbers (can be negative) into an array."""
        return self.read_array(type_code="i", cast=np.int32, length=length)

    def uint8(self) -> int:
        """Read one 8-bit whole number from 0 to 255."""
        return self.read_single(type_code="B", cast=int)

    def uint16(self) -> int:
        """Read one 16-bit whole number from 0 to 65,535."""
        return self.read_single(type_code="H", cast=int)

    def uint32(self) -> int:
        """Read one 32-bit whole number from 0 upward."""
        return self.read_single(type_code="I", cast=int)

    def uint64(self) -> int:
        """Read one 64-bit whole number from 0 upward."""
        return self.read_single(type_code="Q", cast=int)
