"""Tests for src.utils.statistics."""

import numpy as np
import pytest

from src.utils.statistics import compute_statistics, compare_statistics

EXPECTED_KEYS = {"mean", "std", "skewness", "kurtosis", "min", "max", "count"}


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, 300)


def test_compute_statistics_keys(returns):
    stats = compute_statistics(returns)
    assert set(stats.keys()) == EXPECTED_KEYS


def test_compute_statistics_count(returns):
    stats = compute_statistics(returns)
    assert stats["count"] == len(returns)


def test_compare_statistics_columns():
    rng = np.random.default_rng(0)
    d = {"A": rng.normal(0, 0.01, 200), "B": rng.normal(0, 0.02, 200)}
    df = compare_statistics(d)
    assert list(df.columns) == ["A", "B"]
    assert set(df.index) == EXPECTED_KEYS


# TODO: add tests for src.utils.plotting once a non-interactive backend is
#       configured in CI (e.g. matplotlib.use("Agg") in conftest.py).
