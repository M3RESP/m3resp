"""Fixtures for the real-data tests: find the recordings and run the
workflows once per test session."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_data_helpers as helpers


@pytest.fixture(scope="session")
def testps3_paths() -> dict[str, str]:
    """Paths of the dataset 3 files; skips the test when one is missing."""

    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    paths, reason = helpers.dataset_paths("testps3")
    if paths is None:
        pytest.skip(reason)
    return paths


@pytest.fixture(scope="session")
def testps3_run(testps3_paths: dict[str, str]) -> dict:
    """All workflows of dataset 3, run once (see `helpers.run_dataset`)."""

    return helpers.run_dataset("testps3", testps3_paths)


@pytest.fixture(scope="session")
def testps3_results(testps3_run: dict) -> dict:
    """The numbers the tests check (see `helpers.collect_results`)."""

    return helpers.collect_results(testps3_run)
