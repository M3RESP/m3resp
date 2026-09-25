"""Reading Biopac/AcqKnowledge tab-delimited .txt exports."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.resurfemg_adapter._shared import _load_biopac_txt

_HEADER = """Paw_EMG.gtl
0.5 msec/sample
3 channels
Paw - TSD104A - Blood Pressure, DA100C
cmH2O
EMGdi - EMG100C
mV
EMGsc - EMG100C
mV
"""

# Channel k holds the values k.1, k.2, k.3, so a shifted column is easy to see.
_ROWS = [(1.1, 2.1, 3.1), (1.2, 2.2, 3.2), (1.3, 2.3, 3.3)]


def _write(tmp_path, with_time_column: bool):
    lines = [_HEADER.rstrip("\n")]
    if with_time_column:
        lines.append("min\tCH1\tCH2\tCH3\t")
        lines.append("\t3\t3\t3\t")
        lines += [
            f"{i * 0.0005 / 60:g}\t" + "\t".join(map(str, row)) + "\t"
            for i, row in enumerate(_ROWS)
        ]
    else:
        lines.append("CH1\tCH2\tCH3\t")
        lines.append("3\t3\t3\t")
        lines += ["\t".join(map(str, row)) + "\t" for row in _ROWS]
    path = tmp_path / ("with_time.txt" if with_time_column else "without_time.txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("with_time_column", [False, True])
def test_each_label_gets_its_own_channel(tmp_path, with_time_column):
    recording = _load_biopac_txt(_write(tmp_path, with_time_column))

    metadata = recording["metadata"]
    assert metadata["fs"] == 2000.0
    assert metadata["labels"] == ["Paw", "EMGdi", "EMGsc"]
    assert metadata["units"] == ["cmH2O", "mV", "mV"]
    np.testing.assert_allclose(recording["array"], np.array(_ROWS).T)


def test_a_leading_time_column_is_skipped_and_named():
    """TestPS-style exports start each row with a time column ('min')."""

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        with_time = _load_biopac_txt(_write(Path(folder), True))
        without_time = _load_biopac_txt(_write(Path(folder), False))

    assert with_time["metadata"]["skipped_time_column"] == "min"
    assert "skipped_time_column" not in without_time["metadata"]
