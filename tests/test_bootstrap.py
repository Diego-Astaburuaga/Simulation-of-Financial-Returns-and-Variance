"""Tests for src.models.bootstrap."""

import numpy as np
import pytest

from src.models.bootstrap import simulate_bootstrap


@pytest.fixture
def historical_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, 500)


def test_output_shape(historical_returns):
    paths = simulate_bootstrap(historical_returns, n_steps=252, n_paths=100, random_state=1)
    assert paths.shape == (100, 252)


def test_reproducibility(historical_returns):
    paths_a = simulate_bootstrap(historical_returns, n_steps=50, n_paths=10, random_state=42)
    paths_b = simulate_bootstrap(historical_returns, n_steps=50, n_paths=10, random_state=42)
    np.testing.assert_array_equal(paths_a, paths_b)


def test_values_drawn_from_pool(historical_returns):
    paths = simulate_bootstrap(historical_returns, n_steps=50, n_paths=20, random_state=7)
    pool = set(historical_returns.tolist())
    for value in paths.ravel():
        assert value in pool
