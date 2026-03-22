"""Geometric Brownian Motion (GBM) simulation (placeholder).

Planned functionality
---------------------
* ``simulate_gbm(S0, mu, sigma, n_steps, n_paths, dt=1/252, random_state=None)``
    Simulate price paths under the standard GBM assumption:

        dS = mu * S * dt + sigma * S * dW

    Returns both the price-level matrix ``S`` of shape ``(n_paths, n_steps+1)``
    and the corresponding log-return matrix.

* ``fit_gbm(prices)``
    Estimate drift ``mu`` and diffusion ``sigma`` from a historical price series
    using maximum-likelihood (equivalent to computing mean and std of log-returns).

Suggested dependencies
----------------------
- NumPy only (no additional packages required).

References
----------
- Black & Scholes (1973) "The Pricing of Options and Corporate Liabilities"
- Hull (2018) "Options, Futures, and Other Derivatives", Chapter 15
"""

# TODO: implement fit_gbm
# TODO: implement simulate_gbm
