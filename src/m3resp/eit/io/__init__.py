"""Read EIT files from disk.

``read_eit_file`` reads one file with the reader of its device maker, and
``join_eit_files`` joins files recorded one after the other.
``read_draeger`` reads a Draeger ``.bin`` file, ``read_sentec`` a Sentec
``.zri`` file and ``read_timpel`` a Timpel ``.txt`` file, each into an
``EITFileData``. ``BinReader`` reads numbers and text from a binary file, one
value or one block of values at a time.
"""

from m3resp.eit.io.binreader import BinReader
from m3resp.eit.io.draeger import read_draeger
from m3resp.eit.io.file_data import EITFileData
from m3resp.eit.io.join import join_eit_files
from m3resp.eit.io.read_file import VENDOR_READERS, read_eit_file
from m3resp.eit.io.sentec import read_sentec
from m3resp.eit.io.timpel import read_timpel

__all__ = [
    "VENDOR_READERS",
    "BinReader",
    "EITFileData",
    "join_eit_files",
    "read_draeger",
    "read_eit_file",
    "read_sentec",
    "read_timpel",
]
