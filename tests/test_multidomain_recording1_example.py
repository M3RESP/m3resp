"""The multidomain Recording1 example gives the multidomain numbers for window A2.

The expected values come from the multidomain results notebook's own chain
(tools/visualization_tools/paper_results_v2.py, adaptive stage off) on the
same window. Skipped when the session recordings are not on disk.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest
import yaml

from m3resp import M3Session
from m3resp.workflows import load_spec, run_pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(
    REPO_ROOT, "examples", "multidomain_recording1", "recording1_a2.pipeline.yaml"
)
DATA = os.path.join(
    REPO_ROOT, "data", "source", "Test_PS_27072026", "EMG_Vent", "TestPS3.txt"
)

EXPECTED_BREATH_TIMES_S = [
    146.1735, 152.2125, 158.251, 164.8705, 171.0465,
    177.0665, 183.5475, 189.7175, 196.196,
]  # fmt: skip


@pytest.mark.skipif(not os.path.isfile(DATA), reason="session recordings not on disk")
def test_recording1_a2_gives_the_multidomain_numbers():
    pytest.importorskip("resurfemg")
    with open(SPEC, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    spec.pop("outputs", None)  # do not write result files from a test

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_pipeline(
            load_spec(spec, root=os.path.dirname(SPEC)), session=M3Session()
        )

    context = result.context
    fs = float(context.get("processed_emg")["fs"])
    peak_times = np.asarray(context.get("peak_indices")) / fs + 140.0
    assert len(context.get("ecg_peak_indices")) == 77
    np.testing.assert_allclose(peak_times, EXPECTED_BREATH_TIMES_S, atol=1e-6)
    assert context.get("respiratory_rate")[0] == pytest.approx(9.719749582217148)
