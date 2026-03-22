"""Bootstrap simulation of financial returns.

Resamples historical returns with replacement to generate multiple
simulated paths. This is a non-parametric approach that preserves the
empirical distribution of the original data.
"""

import numpy as np


def simulate_bootstrap(returns, n_steps, n_paths, random_state=None):
    """Simulate return paths by resampling historical returns with replacement.

    Parameters
    ----------
    returns : array-like of shape (n,)
        Historical return series used as the resampling pool.
    n_steps : int
        Number of time steps in each simulated path.
    n_paths : int
        Number of independent paths to generate.
    random_state : int or None, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    numpy.ndarray of shape (n_paths, n_steps)
        Simulated return paths, where each row is one path.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.normal(0, 0.01, 500)
    >>> paths = simulate_bootstrap(returns, n_steps=252, n_paths=100)
    >>> paths.shape
    (100, 252)
    """
    rng = np.random.default_rng(random_state)
    returns = np.asarray(returns)
    return rng.choice(returns, size=(n_paths, n_steps), replace=True)
