"""Read a Sentec ``.zri`` EIT file.

A Sentec file starts with a version number. Then it holds a row of records.
Each record has a time stamp, a domain number and a number of data fields.
Each data field has a data number, a size and its content (the payload).

The reader keeps three kinds of data field:

- the time stamp of a frame, in microseconds (domain 16, data 0);
- the "zero reference" image of a frame (domain 16, data 5). The sign of
  its values is turned around;
- the sample frequency (domain 64, data 1).

It skips all other data fields. The file holds no waveforms and no markers.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/loading/sentec.py::
                load_from_single_path, _read_data_field, _read_frame, Domain,
                MeasurementDataID, ConfigurationDataID,
                SENTEC_SAMPLE_FREQUENCY
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `load_from_single_path` is `read_sentec`. It returns an `EITFileData`
      with the images as a `Signal`.
    - The path is read with `os.path`.
    - Docstrings and comments are reworded.
    - Bug fix: each image is stored at its place after `first_frame`
      (`index - first_frame`). The image store has room for `max_frames`
      images, and the original stores each image at `index`, which also
      counts the skipped frames. So with `first_frame` above 0 and
      `max_frames` both given, eitprocessing 1.8.7 stops with an IndexError
      once `index` reaches `max_frames`. The fix gives the frames
      `first_frame` up to `first_frame + max_frames`. The results for all
      other options are unchanged.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import mmap
import os
import warnings
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from m3resp.data.signals import Signal
from m3resp.eit.io.binreader import BinReader
from m3resp.eit.io.file_data import EITFileData

VENDOR = "sentec"

#: The sample frequency in Hz when the file does not give one.
SENTEC_SAMPLE_FREQUENCY = 50.2


def read_sentec(
    path: str | os.PathLike[str],
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> EITFileData:
    """Read one Sentec ``.zri`` file.

    The time stamps count microseconds and start again at 0 after 2**32 - 1.
    The reader adds 2**32 - 1 at each such jump, then turns them into
    seconds.

    Args:
        path (str | os.PathLike[str]): The ``.zri`` file.
        sample_frequency (float | None): The sample frequency in Hz. The
            value in the file is used when there is one; a different given
            value then gives a warning. ``None`` uses the value in the file,
            or else 50.2 Hz.
        first_frame (int): The index of the first frame to read, counted
            from 0.
        max_frames (int | None): The largest number of frames to read.
            ``None`` reads up to the end of the file.

    Returns:
        EITFileData: The images (``pixel_impedance``), with time in seconds
            from the clock of the device. ``signals``, ``events`` and
            ``intervals`` are empty.

    Raises:
        ValueError: If ``first_frame`` is at or after the number of frames in
            the file.
        OSError: If the size of an image does not match its width and height.

    Warns:
        RuntimeWarning: If the given sample frequency differs from the value
            in the file, or fewer than ``max_frames`` frames are read.
    """
    path = os.fspath(path)
    time: list[float] = []
    index = 0
    with (
        open(path, "rb") as fo,
        mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh,
    ):
        file_length = os.fstat(fo.fileno()).st_size
        reader = BinReader(fh, endian="little")
        version = reader.uint8()

        max_n_images = int(file_length / 32 / 32 / 4)
        if max_frames is not None:
            max_n_images = min(max_n_images, max_frames)

        image = np.full(shape=(max_n_images, 32, 32), fill_value=np.nan)

        # Read records until the end of the file, or until enough time
        # stamps are read.
        while fh.tell() < file_length and (len(time) < max_n_images):
            _ = reader.uint64()  # the time stamp of the record is not used
            domain_id = reader.uint8()
            number_data_fields = reader.uint8()

            for _ in range(number_data_fields):
                index, sample_frequency = _read_data_field(
                    reader,
                    domain_id,
                    index,
                    first_frame,
                    fh,
                    time,
                    version,
                    image,
                    sample_frequency,
                )

    if first_frame >= index:
        msg = (
            f"`first_frame` ({first_frame}) is larger than or equal to the "
            f"number of frames in the file ({index})."
        )
        raise ValueError(msg)

    image = image[: index - first_frame, :, :]
    n_frames = len(image)

    if max_frames and n_frames != max_frames:
        msg = (
            f"The number of frames requested ({max_frames}) is larger "
            f"than the available number ({n_frames}) of frames after "
            f"the first frame selected ({first_frame}, total frames: "
            f"{index}).\n {n_frames} frames were loaded."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if sample_frequency is None:
        sample_frequency = SENTEC_SAMPLE_FREQUENCY

    time_array = np.unwrap(np.array(time), period=np.iinfo(np.uint32).max) / 1_000_000

    pixel_signal = Signal(
        values=image,
        time=time_array,
        sample_frequency=float(sample_frequency),
        name="raw",
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        source=VENDOR,
    )
    return EITFileData(
        vendor=VENDOR,
        path=path,
        sample_frequency=float(sample_frequency),
        pixel_impedance=pixel_signal,
    )


def _read_data_field(
    reader: BinReader,
    domain_id: int,
    index: int,
    first_frame: int,
    fh: mmap.mmap,
    time: list[float],
    version: int,
    image: np.ndarray,
    sample_frequency: float | None,
) -> tuple[int, float | None]:
    """Read one data field, and return the new frame count and sample frequency.

    Args:
        reader (BinReader): The reader, at the start of the data field.
        domain_id (int): The domain number of the record.
        index (int): The number of images found so far, skipped ones
            included.
        first_frame (int): The index of the first frame to keep. Time stamps
            and images before it are skipped.
        fh (mmap.mmap): The open file, used to skip data.
        time (list[float]): Time stamps in microseconds; a new one is added
            here.
        version (int): The file version.
        image (np.ndarray): The image store; ``image[index - first_frame]``
            is set.
        sample_frequency (float | None): The sample frequency so far, in Hz.

    Returns:
        tuple[int, float | None]: The frame count and the sample frequency
            after this data field.

    Raises:
        OSError: If the size of an image does not match its width and height.

    Warns:
        RuntimeWarning: If the sample frequency in the file differs from the
            given one.
    """
    data_id = reader.uint8()
    payload_size = reader.uint16()

    if payload_size == 0:
        return index, sample_frequency

    match domain_id, data_id:
        case Domain.MEASUREMENT, MeasurementDataID.TIMESTAMP:
            if index < first_frame:
                fh.seek(payload_size, os.SEEK_CUR)
            else:
                time.append(reader.uint32())

        case Domain.MEASUREMENT, MeasurementDataID.ZERO_REF_IMAGE:
            if index < first_frame:
                fh.seek(payload_size, os.SEEK_CUR)
            else:
                ref = _read_frame(
                    version=version, payload_size=payload_size, reader=reader
                )
                # Bug fix (see Provenance): the place after first_frame.
                image[index - first_frame, :, :] = ref

            index += 1

        case Domain.CONFIGURATION, ConfigurationDataID.SAMPLE_FREQUENCY:
            loaded_sample_frequency = np.round(reader.float32(), 4)
            if sample_frequency and not np.isclose(
                loaded_sample_frequency, sample_frequency
            ):
                msg = (
                    f"Sample frequency provided ({sample_frequency:.2f} Hz) "
                    f"differs from value found in file "
                    f"({loaded_sample_frequency:.2f} Hz). "
                    f"The sample frequency value found in the file will be used."
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            sample_frequency = loaded_sample_frequency

        case _, _:
            fh.seek(payload_size, os.SEEK_CUR)

    return index, sample_frequency


def _read_frame(
    version: int,
    payload_size: int,
    reader: BinReader,
) -> NDArray:
    """Read one image.

    From version 2 on, the image starts with a quality number, which is
    skipped. Then come the width, the height and the pixel values.

    Args:
        version (int): The file version.
        payload_size (int): The size of the data field in bytes.
        reader (BinReader): The reader, at the start of the image.

    Returns:
        NDArray: The image, with the sign of the values turned around, in
            the shape (width, height).

    Raises:
        OSError: If the number of pixels is not width times height.
    """
    if version > 1:
        _ = reader.uint8()  # quality number, not used
        n_pixels = (payload_size - 3) // 4
    else:
        n_pixels = (payload_size - 2) // 4

    image_width = reader.uint8()
    image_height = reader.uint8()

    if image_width * image_height != n_pixels:
        msg = (
            f"The length of image array is {n_pixels} which is not equal to "
            f"the product of the width ({image_width}) and height "
            f"({image_height}) of the frame."
        )
        raise OSError(msg)

    # The zero reference values are stored with the opposite sign.
    zero_ref = -reader.npfloat32(n_pixels)
    return np.reshape(zero_ref, (image_width, image_height), order="C")


class Domain(IntEnum):
    """The domain number of a record."""

    MEASUREMENT = 16
    CONFIGURATION = 64


class MeasurementDataID(IntEnum):
    """The data number of a measurement data field."""

    TIMESTAMP = 0
    ZERO_REF_IMAGE = 5


class ConfigurationDataID(IntEnum):
    """The data number of a configuration data field."""

    SAMPLE_FREQUENCY = 1
