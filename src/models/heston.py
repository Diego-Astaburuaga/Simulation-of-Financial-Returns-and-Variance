"""Heston stochastic volatility model (placeholder).

Planned functionality
---------------------
* ``fit_heston(returns, dt=1/252, method='scipy')``
    Calibrate Heston parameters (kappa, theta, xi, rho, v0) from a return
    series, e.g. via moment-matching or characteristic-function MLE.

* ``simulate_heston(params, S0, n_steps, n_paths, dt=1/252, random_state=None)``
    Simulate correlated price and variance paths using the Euler-Maruyama
    (or Milstein) discretisation of:

        dS = mu * S * dt + sqrt(v) * S * dW_S
        dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
        dW_S * dW_v = rho * dt

    Returns a dict with ``'prices'`` and ``'variances'`` arrays of shape
    ``(n_paths, n_steps+1)``.

Suggested dependencies
----------------------
- NumPy for discretisation.
- SciPy for calibration optimisation.

References
----------
- Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility"
- Andersen (2007) "Efficient Simulation of the Heston Stochastic Volatility Model"
"""

# TODO: implement fit_heston
# TODO: implement simulate_heston
