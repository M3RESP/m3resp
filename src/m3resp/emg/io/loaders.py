"""Loaders for various EMG data formats."""  # noqa: CPY001

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from m3resp.core.exceptions import OptionalDependencyError


class _LoaderMixin:
    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    def load(self, file_path: str, **kwargs) -> dict[str, Any]:
        """Load a file as numpy array.

        This function loads a file from a given path and returns the data as a
        numpy array. The function can handle .poly5, .mat, .csv, and .npy files.
        The function can also rename channels and drop channels from the data.

        Args:
            file_path (str): Path to the file to be loaded.
            verbose (bool): Print verbose output.
            **kwargs: Additional keyword arguments for specific file loaders:

                - key_name (str): Key name for loading .mat files.
                - force_col_reading (bool): If True, force reading columns for
                .csv files. Default is False.
                - record_index (int): Record index for loading .adi* files.
                Default is 0.
                - channel_indexes (list): List of channel indices for loading .adi
                files.
                - labels (list): List of new channel names to rename the columns.

        Returns:
            dict:
                - numpy.ndarray: Numpy array of the loaded data.
                - pandas.DataFrame: Pandas DataFrame of the loaded data.
                - dict: Metadata of the loaded data.

        Raises:
            TypeError: If file_path is not a str.
        """
        if not isinstance(file_path, str):
            msg = "file_path should be a str."
            raise TypeError(msg)
        if self._loader is not None:
            return self._loader(file_path, **kwargs)
        file_extension = Path(file_path).name.split(".")[-1].lower()
        loaders = {
            "poly5": _load_poly5,
            "mat": lambda fp: _load_mat(fp, kwargs.get("key_name", "")),
            "csv": lambda fp: _load_csv(fp, kwargs.get("force_col_reading", False)),
            "npy": _load_npy,
            "adi": lambda fp: _load_adicht(
                fp,
                kwargs.get("record_index", 0),
                kwargs.get("channel_indexes"),
                kwargs.get("resample_channels"),
            ),
        }
        if file_extension.startswith("adi"):
            file_extension = "adi"
        if file_extension in loaders:
            return loaders[file_extension](file_path)

        msg = f"No methods available for file extension {file_extension}."
        raise UserWarning(msg)


def _load_biopac_txt(path: str) -> dict[str, Any]:
    """Load a Biopac/AcqKnowledge tab-delimited ``.txt`` export.

    The header looks like::
        Paw_EMG.gtl
        0.5 msec/sample
        3 channels
        Paw - TSD104A - Blood Pressure, DA100C
        cmH2O
        EMGdi - EMG100C
        mV
        EMGps - EMG100C
        mV
        CH1<TAB>CH2<TAB>CH3
        3571881<TAB>3571887<TAB>3571887
        <numeric data rows...>

    i.e. a title line, a ``msec/sample`` sampling-interval line, a
    ``N channels`` line, then two lines per channel (label, unit), a ``CHn``
    column header, a per-channel sample-count row, and finally the samples.
    Returns the same ``(array, dataframe, metadata)``-shaped dict as
    :meth:`ReSurfEMGAdapter.load`, with ``array`` channel-major
    ``(n_channels, n_samples)`` and ``metadata["fs"]`` populated.
    """
    with Path.open(Path(path), encoding="utf-8", errors="replace") as handle:
        header: list[str] = [handle.readline().rstrip("\n") for _ in range(3)]

    recording_name = header[0].strip()
    time_delta, time_unit = float(header[1].split()[0]), header[1].split()[1]
    fs = 1000.0 / time_delta if time_unit.lower() == "msec/sample" else 1.0 / time_delta
    n_channels = int(header[2].split()[0])

    labels: list[str] = []
    units: list[str] = []
    with Path.open(Path(path), encoding="utf-8", errors="replace") as handle:
        for _ in range(3):
            handle.readline()
        for _ in range(n_channels):
            label_line = handle.readline().rstrip("\n")
            unit_line = handle.readline().rstrip("\n")
            # "Paw - TSD104A - Blood Pressure, DA100C" -> "Paw"
            labels.append(label_line.split(" - ")[0].strip())
            units.append(unit_line.strip())

    # 3 title/rate/channel lines + 2 lines per channel + column-header row
    # + per-channel sample-count row precede the numeric samples.
    skiprows = 3 + 2 * n_channels + 2
    dataframe = pd.read_csv(
        path,
        sep="\t",
        skiprows=skiprows,
        names=labels,
        usecols=range(n_channels),
        engine="c",
    )
    array = dataframe.to_numpy(dtype=float).T  # channel-major (n_channels, n_samples)
    metadata = {
        "fs": fs,
        "labels": labels,
        "units": units,
        "file_name": Path(path).name,
        "file_dir": str(Path(path).parent),
        "file_extension": "txt",
        "recording_name": recording_name,
    }
    return {"array": array, "dataframe": dataframe, "metadata": metadata}


def _load_poly5(file_path: str) -> dict[str, Any]:
    """Load a .Poly5 file and return the data as a pandas DataFrame.

    This function loads a .Poly5 file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        dict:
            - array (np.ndarray): Numpy array of the loaded data.
            - dataframe (pandas.DataFrame): Pandas DataFrame of the loaded data.
            - metadata (dict): Metadata of the loaded data.
    """
    from .vendors.poly5reader import Poly5Reader  # noqa: PLC0415

    poly5_data = Poly5Reader(file_path)
    n_samples = poly5_data.num_samples
    loaded_data = poly5_data.samples[:, :n_samples]
    metadata = {}
    metadata["fs"] = poly5_data.sample_rate
    metadata["labels"] = list(poly5_data.ch_names)
    metadata["units"] = list(poly5_data.ch_unit_names)
    array = np.asarray(loaded_data)
    dataframe = pd.DataFrame(array.T, columns=metadata["labels"])

    return {"array": array, "dataframe": dataframe, "metadata": metadata}


def _load_mat(file_path: str, key_name: str | None = None) -> dict[str, Any]:
    """Load a .mat file and return the data as a pandas DataFrame.

    This function loads a .mat file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
        file_path (str): Path to the file to be loaded.
        key_name (str): Key name for .mat files.

    Returns:
        dict:
            - array (np.ndarray): Numpy array of the loaded data.
            - dataframe (pandas.DataFrame): Pandas DataFrame of the loaded data.
            - metadata (dict): Metadata of the loaded data.

    Raises:
        TypeError: If no key_name is provided.
    """
    sio = _sio()
    mat_dict = sio.loadmat(file_path, mdict=None, appendmat=False)
    array = np.array([])
    dataframe = pd.DataFrame()

    if isinstance(key_name, str):
        array = mat_dict[key_name]
        if array.shape[0] > array.shape[1]:
            array = np.rot90(array)
        dataframe = pd.DataFrame(array.T)

    else:
        msg = "No key_name provided."
        raise TypeError(msg)

    return {"array": array, "dataframe": dataframe, "metadata": {}}


# TODO: cleanup
def _load_csv(
    file_path: str,
    force_col_reading: bool,  # verbose: bool = True
) -> dict[str, Any]:
    """Load a .csv file and return the data as a pandas DataFrame.

    This function loads a .csv file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the loaded channels.

    Args:
        file_path (str): Path to the file to be loaded.
        force_col_reading (bool): Force column reading for row based .csv
            files.

    Returns:
        dict:
            - array (np.ndarray): Numpy array of the loaded data.
            - dataframe (pandas.DataFrame): Pandas DataFrame of the loaded data.
            - metadata (dict): Metadata of the loaded data.

    Raises:
        UserWarning: If the .csv is row based and force_col_reading is not
            True.
    """
    dataframe = pd.DataFrame()

    def _has_header(file_path: str, nrows: int = 20) -> bool:
        dataframe = pd.read_csv(file_path, header=None, nrows=nrows)
        df_header = pd.read_csv(filepath_or_buffer=file_path, nrows=nrows)
        return tuple(dataframe.dtypes) != tuple(df_header.dtypes)

    def _chech_row_wise(file_path: str, nrows: int = 20) -> bool:
        with Path(file_path).open("r") as f:
            n_lines = sum(1 for _ in f)

        with Path(file_path).open("r") as f:
            col_lg_row = 0
            i = 0
            for i, line in enumerate(f):
                if len(line) > n_lines:
                    col_lg_row += 1
                if i > nrows or col_lg_row > nrows:
                    break
            return not (col_lg_row > nrows // 2 or col_lg_row == n_lines)

    row_wise = _chech_row_wise(file_path, nrows=20)
    if (row_wise is False) and (force_col_reading is not True):
        msg = [
            "The provided .csv is row based. ",
            "This could yield significant loading durations.",
            "If you want to proceed, set force_col_reading=True",
        ]
        raise UserWarning(msg)

    metadata = {}

    if row_wise and _has_header(file_path):
        dataframe = pd.read_csv(file_path)
        array = dataframe.to_numpy()
        metadata["labels"] = dataframe.columns.values
    else:
        dataframe = pd.read_csv(file_path, header=None)
        array = dataframe.to_numpy()

    return {"array": array, "dataframe": dataframe, "metadata": metadata}


def _load_adicht(
    file_path: str,
    record_index: int,
    channel_indexes: list[int] | None = None,
    resample_channels: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Load a .adicht file and return the data as a pandas DataFrame.

    This function loads a .adicht file and returns the data as a pandas
    DataFrame.

    Args:
        file_path (str): Path to the file to be loaded.
        record_index (int): The record index to extract data from.
        channel_indexes (list[int], optional): List of channel indices to
            extract.
        resample_channels (dict[int, int], optional): Map of channel_idx to
            target sampling rate.

    Returns:
        dict:
            - array (np.ndarray): Numpy array of the loaded data.
            - dataframe (pandas.DataFrame): Pandas DataFrame of the loaded data.
            - metadata (dict): Metadata of the loaded data, including sampling rate,
                channel indexes, labels, units, and record ID.

    Raises:
        UserWarning: If multiple sampling rates are detected and no
            channel_indexes are provided.

    """
    from .vendors.adinstruments import AdichtReader  # noqa: PLC0415

    adi_data = AdichtReader(file_path)

    adi_metadata = adi_data.generate_metadata()
    if channel_indexes is None:
        channel_indexes = list(adi_data.channel_map.keys())
    fs_sel = [adi_metadata[idx]["fs"][0] for idx in channel_indexes]
    if len(set(fs_sel)) > 1:
        msg = (
            "Multiple sampling rates detected, which cannot be parsed into "
            "one numpy array. Please specify the channel_indexes to select a subset "
            "of channels with the same sampling rate or resample channels "
            "with the resample_channels argument."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

        max_fs = max(fs_sel)
        channel_indexes = [i for i, fs in enumerate(fs_sel) if fs == max_fs]

    data_df, sampling_frequency = adi_data.extract_data(
        channel_idxs=channel_indexes,
        record_idx=record_index,
        resample_channels=resample_channels,
    )
    metadata = {
        "fs": sampling_frequency,
        "channel_idxs": channel_indexes,
        "labels": adi_data.get_labels(channel_indexes),
        "units": adi_data.get_units(channel_indexes, record_index),
        "record_id": record_index,
    }

    return {"array": data_df.to_numpy(), "dataframe": data_df, "metadata": metadata}


def _load_npy(file_path: str) -> dict[str, Any]:
    """This function loads a .npy file and returns the data as a numpy array.

    Args:
        file_path (str): Path to the file to be loaded.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.
    """
    array = np.load(file_path)
    if array.shape[0] > array.shape[1]:
        array = np.rot90(array)

    dataframe = pd.DataFrame(array)

    return {"array": array, "dataframe": dataframe, "metadata": {}}


def _sio() -> Any:  # noqa: ANN401
    """Import scipy.io and raise an error if not installed."""
    try:
        import scipy.io as sio  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "scipy is required for loading .mat files. "
            "Please install scipy and try again."
        )
        raise OptionalDependencyError(msg) from e
    return sio
