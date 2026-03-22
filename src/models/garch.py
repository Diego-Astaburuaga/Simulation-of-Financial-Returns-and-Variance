"""GARCH / HAR volatility models (placeholder).

Planned functionality
---------------------
* ``fit_garch(returns, p=1, q=1)``
    Fit a GARCH(p, q) model to a return series and return estimated
    parameters (omega, alpha, beta).

* ``simulate_garch(params, n_steps, n_paths, random_state=None)``
    Generate return paths whose conditional variance follows the fitted
    GARCH dynamics.

* ``fit_har(realized_vol, lags=(1, 5, 22))``
    Fit a HAR (Heterogeneous Autoregressive) model to a realized-volatility
    series using OLS.

* ``forecast_har(params, realized_vol, horizon=1)``
    Produce multi-step volatility forecasts from a fitted HAR model.

Suggested dependencies
----------------------
- ``arch`` package for GARCH estimation (``pip install arch``).
- ``statsmodels`` for HAR OLS.

References
----------
- Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
- Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
"""

# TODO: implement fit_garch
# TODO: implement simulate_garch
# TODO: implement fit_har
# TODO: implement forecast_har
