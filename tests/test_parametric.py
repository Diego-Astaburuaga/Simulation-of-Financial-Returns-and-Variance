"""Tests for src.models.parametric."""

import numpy as np
import pytest

from src.models.parametric import fit_normal, fit_student_t, simulate_parametric


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, 500)


def test_fit_normal_keys(returns):
    params = fit_normal(returns)
    assert params["distribution"] == "normal"
    assert "mu" in params and "sigma" in params


def test_fit_student_t_keys(returns):
    params = fit_student_t(returns)
    assert params["distribution"] == "student_t"
    assert "df" in params and "mu" in params and "sigma" in params


def test_simulate_parametric_normal_shape(returns):
    params = fit_normal(returns)
    paths = simulate_parametric(params, n_steps=252, n_paths=100, random_state=0)
    assert paths.shape == (100, 252)


def test_simulate_parametric_student_t_shape(returns):
    params = fit_student_t(returns)
    paths = simulate_parametric(params, n_steps=252, n_paths=100, random_state=0)
    assert paths.shape == (100, 252)


def test_simulate_parametric_unknown_distribution():
    with pytest.raises(ValueError, match="Unknown distribution"):
        simulate_parametric({"distribution": "cauchy"}, n_steps=10, n_paths=5)
