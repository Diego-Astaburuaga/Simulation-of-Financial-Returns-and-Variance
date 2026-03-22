"""Generic Monte Carlo simulation engine.

Provides a thin wrapper that calls any return-simulator and accumulates
the results into a price-path (or raw return) matrix.
"""

import numpy as np


def run_monte_carlo(simulator, n_steps, n_paths, **kwargs):
    """Run a Monte Carlo simulation using the provided simulator function.

    The *simulator* must accept at least ``n_steps`` and ``n_paths`` as
    keyword arguments and return an array of shape ``(n_paths, n_steps)``
    containing simulated **returns** (not price levels).

    The function converts the return matrix to cumulative log-price paths
    by computing the cumulative sum of returns along the time axis, which
    is equivalent to the log-price relative to the start.

    Parameters
    ----------
    simulator : callable
        A function with signature
        ``simulator(n_steps, n_paths, **kwargs) -> ndarray``.
        Examples: :func:`src.models.bootstrap.simulate_bootstrap`,
        :func:`src.models.parametric.simulate_parametric`.
    n_steps : int
        Number of time steps in each simulated path.
    n_paths : int
        Number of independent paths to generate.
    **kwargs
        Additional keyword arguments forwarded verbatim to *simulator*.

    Returns
    -------
    dict with keys:

    ``'returns'`` : numpy.ndarray of shape (n_paths, n_steps)
        Raw simulated returns produced by *simulator*.
    ``'cum_returns'`` : numpy.ndarray of shape (n_paths, n_steps)
        Cumulative sum of returns (log-price relative to time 0).

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.bootstrap import simulate_bootstrap
    >>> historical = np.random.normal(0, 0.01, 500)
    >>> result = run_monte_carlo(
    ...     simulate_bootstrap,
    ...     n_steps=252,
    ...     n_paths=100,
    ...     returns=historical,
    ...     random_state=0,
    ... )
    >>> result['returns'].shape
    (100, 252)
    >>> result['cum_returns'].shape
    (100, 252)
    """
    simulated_returns = simulator(n_steps=n_steps, n_paths=n_paths, **kwargs)
    cum_returns = np.cumsum(simulated_returns, axis=1)
    return {"returns": simulated_returns, "cum_returns": cum_returns}
