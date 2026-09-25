"""Load EMG data from ADInstrument devices.

The class AdichtReader is designed to load EMG data from an ADInstruments
device using the .adicht file format (Labchart) and prepares it for use in
ReSurfEMG. The foundation of the AdichtReader class is the repository
"adinstruments_sdk_python" by Jim Hokanson, available at:
https://github.com/JimHokanson/adinstruments_sdk_python

An example of how to use this class is provided in the main block of this file.
This example executes only if the script is run directly by the Python
interpreter and not when imported as a module.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from m3resp.core.utilities import _validate_incompatible_kwargs

KT = TypeVar("KT")
VT = TypeVar("VT")

if platform.system() == "Windows":
    try:
        import adi
    except ImportError as e:
        msg = (
            "The 'adi' module is required for AdichtReader but could not be "
            "imported. Ensure that the 'adinstruments_sdk_python' package is "
            "installed and available in your Python environment."
        )
        raise ImportError(msg) from e
else:
    msg = "AdichtReader is only available on Windows."
    raise ImportError(msg)


class AdichtReader:
    """Load EMG data from ADInstrument devices.

    Class for loading timeseries data from an ADInstruments devices using
    the .adicht/.adidat/.adibin file formats (LabChart, BIOPAC) and prepare it
    for use in m3resp.
    Based on the "adinstruments_sdk_python" repository by Jim Hokanson,
    available at: https://github.com/JimHokanson/adinstruments_sdk_python
    """

    def __init__(self, file_path: str):
        """Initialize the AdichtReader with the provided file path.

        Args:
                file_path (str): The file path to the import.
        """
        if platform.system() != "Windows":
            msg = "AdichtReader is only available on Windows."
            raise ImportError(msg)
        self.file_path = file_path
        self.metadata: list[dict] = []
        self.metadata_table = None
        self.channel_map: dict[int, int] = {}  # Dictionary mapping channel names to IDs
        self.adicht_data: Any = None  # Reader object for the file
        self.record_map: dict[int, int] = {}  # Dictionary mapping record idx to IDs

        self._validate_file_path()
        self._initialize_reader()
        self._initialize_channel_map()
        self._initialize_record_map()

    def _validate_file_path(self) -> None:
        """Validate the provided file path.

        Validates whether the provided file path exists and is readable.
        """
        if not Path(self.file_path).exists():
            msg = f"The file '{self.file_path}' was not found."
            raise FileNotFoundError(msg)
        if not Path(self.file_path).is_file():
            msg = f"The path '{self.file_path}' does not refer to a file."
            raise ValueError(msg)

    def _initialize_reader(self) -> None:
        """Initialize the ADInstruments reader.

        Initializes the adi-reader and loads the file.
        """
        try:
            self.adicht_data = adi.read_file(  # pyright: ignore[reportPossiblyUnboundVariable]
                self.file_path
            )
        except Exception as e:
            msg = f"Error loading the file: {e}"
            raise RuntimeError(msg) from e

    def _initialize_channel_map(self) -> None:
        """Map channel names to their IDs.

        Creates a dictionary mapping the channel names to their IDs.
        """
        self.channel_map = {
            i: channel.id for i, channel in enumerate(self.adicht_data.channels)
        }

    def _initialize_record_map(self) -> None:
        """Map record indices to their IDs.

        Creates a dictionary mapping the record indices to their IDs.
        """
        self.record_map = {
            i: record.id for i, record in enumerate(self.adicht_data.records)
        }

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"

    def generate_metadata(self) -> list[dict]:
        """Extract channel metadata.

        Extracts metadata on channels, samples, records, sampling rates, units,
        and time step and sets it in self.metadata and self.metadata_table.

        Returns:
            list[dict]: List of metadata dicts per channel.
        """
        _metadata_table = PrettyTable()
        _metadata_table.field_names = [
            "idx",
            "Channel ID",
            "Name",
            "Records",
            "Samples",
            "Sampling Rate (Hz)",
            "timestep (s)",
            "Units",
        ]
        _metadata_table.align["Name"] = "l"

        channel_info = []
        for channel_index, channel in enumerate(self.adicht_data.channels):
            info = {
                "idx": channel_index,
                "id": channel.id,
                "name": channel.name,
                "records": channel.n_records,
                "samples": channel.n_samples,
                "fs": channel.fs,
                "time_step": channel.dt,
                "units": channel.units,
            }
            channel_info.append(info)
            _metadata_table.add_row(
                [
                    channel_index,
                    channel.id,
                    channel.name,
                    channel.n_records,
                    ", ".join(map(str, channel.n_samples)),
                    ", ".join(map(str, channel.fs)),
                    ", ".join(map(str, channel.dt)),
                    channel.units,
                ]
            )
        self.metadata = channel_info
        self.metadata_table = _metadata_table
        return channel_info

    def print_metadata(self) -> None:
        """Print channel metadata.

        Extracts and provides a tabular overview of the channels, samples,
        records, sampling rates, units, and time step.
        """
        self.generate_metadata()
        print(f"Available channels and metadata:\n{self.metadata_table}")  # noqa: T201

    def _resolve_one(
        self, idx: int | None, id_: int | None, mapping: dict[int, int], name: str
    ) -> int:
        if idx is not None:
            return idx
        if id_ is None:
            msg = f"Either {name}_idx or {name}_id must be set."
            raise ValueError(msg)
        resolved = _get_key_from_value(mapping, id_)
        if resolved is None:
            msg = f"{name} id {id_} not found."
            raise ValueError(msg)
        return resolved

    def _resolve_many(
        self,
        idxs: list[int] | None,
        ids: list[int] | None,
        mapping: dict[int, int],
        name: str,
    ) -> list[int]:
        if idxs is not None:
            return idxs
        if ids is None:
            msg = f"Either {name}_idxs or {name}_ids must be set."
            raise ValueError(msg)
        return [self._resolve_one(None, i, mapping, name) for i in ids]

    def get_labels(
        self, channel_indexes: list[int] | None = None, channel_ids: list | None = None
    ) -> list[str]:
        """Return channel names based on channel indices or IDs.

        Args:
                channel_indexes (list[int], optional): List of channel indices.
                channel_ids (list, optional): List of channel IDs. Either
                    channel_indexes or channel_ids must be set.

        Returns:
                list[str]: List of channel names.
        """
        _channel_indexes = self._resolve_many(
            channel_indexes, channel_ids, self.channel_map, "channel"
        )
        return [self.adicht_data.channels[index].name for index in _channel_indexes]

    def get_units(
        self,
        channel_indexes: list[int] | None = None,
        record_index: int | None = None,
        channel_ids: list | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        """Return channel units based on channel indices and a record index or ID.

        Args:
                channel_indexes (list[int], optional): List of channel indices. Either
                    channel_indexes or channel_ids must be set.
                channel_ids (list, optional): List of channel IDs. Either
                    channel_indexes or channel_ids must be set.
                record_index (int, optional): The record index to retrieve the units
                    for. Either record_index or record_id must be set.
                record_id (int, optional): The record ID to retrieve the units for.
                    Either record_index or record_id must be set.

        Returns:
                list[str]: List of units.
        """
        _channel_indexes = self._resolve_many(
            channel_indexes, channel_ids, self.channel_map, "channel"
        )
        _record_index = self._resolve_one(
            record_index, record_id, self.record_map, "record"
        )
        return [
            self.adicht_data.channels[idx].units[_record_index]
            for idx in _channel_indexes
        ]

    def resample_channel(
        self,
        fs_target: int,
        channel_index: int | None = None,
        record_index: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Resample a channel to a target sampling rate.

            Resamples the specified channel using linear interpolation.


        Args:
                fs_target (int): The target sampling rate in Hz.
                channel_index (int, optional): The channel index to be resampled.
                record_index (int, optional): The record index to be resampled.
                    Either record_index or record_id must be set.
                **kwargs: Additional arguments to specify channel_id or record_id
                    instead of indices.

        Returns:
                pd.DataFrame: Record DataFrame with resampled data for the
                    specified index.
        """
        _channel_index = _validate_incompatible_kwargs(
            "channel_index",
            "channel_id",
            arg_value1=channel_index,
            default_value=None,
            **kwargs,
        )
        if _channel_index is None:
            msg = "Either channel_index or channel_id must be set."
            raise ValueError(msg)
        _record_index = _validate_incompatible_kwargs(
            "record_index",
            "record_id",
            arg_value1=record_index,
            default_value=None,
            **kwargs,
        )
        if _record_index is None:
            msg = "Either record_index or record_id must be set."
            raise ValueError(msg)

        _channel = self.adicht_data.channels[_channel_index]
        if fs_target == 1 / _channel.dt[_record_index]:
            msg = "target_rate equals current_rate"
            raise UserWarning(msg)

        # Create DataFrame and set time index
        df = pd.DataFrame(
            {_channel.name: _channel.get_data(self.record_map[_record_index])}
        )
        df.index = pd.to_timedelta(df.index * _channel.dt[_record_index], unit="s")

        # New interval based on target rate
        dt_target_timedelta = pd.to_timedelta(1 / fs_target, unit="s")

        fs_original = _channel.fs[_record_index]
        n_samples_target = int(
            _channel.n_samples[_record_index] * (fs_target / fs_original)
        )

        # Create an empty DataFrame with target sample rate
        timedelta_index = pd.to_timedelta(
            np.arange(n_samples_target) * dt_target_timedelta.value
        )
        empty_df = pd.DataFrame(index=timedelta_index, columns=[_channel.name])
        empty_df[_channel.name] = np.nan

        # Merge DataFrames
        df_combined = empty_df.combine_first(df)
        df_combined = df_combined.interpolate(method="linear")
        return df_combined.resample(dt_target_timedelta).interpolate(method="linear")

    def extract_data(
        self,
        channel_index: list[int] | None = None,
        record_index: int | None = None,
        resample_channels: dict[int, int] | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, int]:
        """Extract channel data from specified channels and record.

            Optionally resamples specified channels to equalize sampling rates
            across channels. Resampling all channels to a rate not yet used is not
            supported; at least one channel must already have the target rate and
            must not be listed in resample_channels.


        Args:
                channel_index (list[int], optional): List of channel indices.
                record_index (int, optional): The record index to extract data from.
                resample_channels (dict[int, int], optional): Map of
                    channel_idx to target rate. Example: ``{1: 2000, 3: 2000}``
                    resamples channels 1 and 3 to 2000 Hz.
                **kwargs: Additional arguments to specify channel_ids or record_id
                    instead of indices.

        Returns:
                tuple:
                    - pd.DataFrame: Extracted (and optionally resampled) data.
                    - int: Sampling rate (Hz) of the leading channel.
        """
        _channel_index = self._resolve_many(
            channel_index, kwargs.get("channel_ids"), self.channel_map, "channel"
        )
        _record_index = self._resolve_one(
            record_index, kwargs.get("record_id"), self.record_map, "record"
        )

        fs_out = []
        data_dict = {}
        non_resampled_channels = []
        for idx in _channel_index:
            if idx not in self.channel_map:
                msg = f"Channel idx '{idx}' is invalid."
                raise ValueError(msg)

            if resample_channels and idx in resample_channels:
                resampled_df = self.resample_channel(
                    resample_channels[idx], idx, _record_index
                )

                for column in resampled_df.columns:
                    data_dict[column] = resampled_df[column].values
                fs_out.append(resample_channels[idx])
            else:
                np_data = self.adicht_data.channels[idx].get_data(
                    self.record_map[_record_index]
                )
                channel_name = self.adicht_data.channels[idx].name
                data_dict[channel_name] = np_data
                non_resampled_channels.append(idx)
                fs_out.append(self.metadata[idx]["fs"][0])

        if len(set(fs_out)) > 1:
            msg = "Output channels have different sampling rates."
            raise ValueError(msg)
        df = pd.DataFrame(data_dict)
        # Select an unsampled channel to read out target sampling
        leader_channel = self.adicht_data.channels[non_resampled_channels[0]]
        df.index = pd.to_timedelta(
            df.index * leader_channel.dt[_record_index], unit="s"
        )

        return df, int(leader_channel.fs[_record_index])


def _get_key_from_value[KT, VT](dictionary: dict[KT, VT], value: VT) -> KT | None:
    """Return the key of a dictionary where the value matches the input value.

    Args:
        dictionary (dict): Dictionary to search.
        value: Value to search for.

    Returns:
        Key where value is found, or None if not found.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
