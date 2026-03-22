"""Tests for src.simulation.monte_carlo."""

import numpy as np
import pytest

from src.models.bootstrap import simulate_bootstrap
from src.simulation.monte_carlo import run_monte_carlo


@pytest.fixture
def historical_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, 500)


def test_result_keys(historical_returns):
    result = run_monte_carlo(
        simulate_bootstrap,
        n_steps=50,
        n_paths=20,
        returns=historical_returns,
        random_state=0,
    )
    assert "returns" in result
    assert "cum_returns" in result


def test_result_shapes(historical_returns):
    result = run_monte_carlo(
        simulate_bootstrap,
        n_steps=50,
        n_paths=20,
        returns=historical_returns,
        random_state=0,
    )
    assert result["returns"].shape == (20, 50)
    assert result["cum_returns"].shape == (20, 50)


def test_cum_returns_is_cumsum(historical_returns):
    result = run_monte_carlo(
        simulate_bootstrap,
        n_steps=30,
        n_paths=10,
        returns=historical_returns,
        random_state=1,
    )
    expected = np.cumsum(result["returns"], axis=1)
    np.testing.assert_array_equal(result["cum_returns"], expected)
