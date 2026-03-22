"""Parametric simulation of financial returns.

Fits Normal or Student-t distributions to historical returns and
generates simulated paths by drawing from the fitted distributions.
"""

import numpy as np
from scipy import stats


def fit_normal(returns):
    """Fit a Normal distribution to historical returns.

    Parameters
    ----------
    returns : array-like of shape (n,)
        Historical return series.

    Returns
    -------
    dict
        Dictionary with keys ``'distribution'``, ``'mu'``, and ``'sigma'``.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.normal(0, 0.01, 500)
    >>> params = fit_normal(returns)
    >>> params['distribution']
    'normal'
    """
    returns = np.asarray(returns)
    mu, sigma = stats.norm.fit(returns)
    return {"distribution": "normal", "mu": mu, "sigma": sigma}


def fit_student_t(returns):
    """Fit a Student-t distribution to historical returns.

    Parameters
    ----------
    returns : array-like of shape (n,)
        Historical return series.

    Returns
    -------
    dict
        Dictionary with keys ``'distribution'``, ``'df'``, ``'mu'``,
        and ``'sigma'``.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.standard_t(5, size=500) * 0.01
    >>> params = fit_student_t(returns)
    >>> params['distribution']
    'student_t'
    """
    returns = np.asarray(returns)
    df, mu, sigma = stats.t.fit(returns)
    return {"distribution": "student_t", "df": df, "mu": mu, "sigma": sigma}


def simulate_parametric(params, n_steps, n_paths, random_state=None):
    """Simulate return paths from a fitted parametric distribution.

    Accepts the parameter dictionaries produced by :func:`fit_normal` or
    :func:`fit_student_t`.

    Parameters
    ----------
    params : dict
        Fitted distribution parameters.  Must contain a ``'distribution'``
        key equal to ``'normal'`` or ``'student_t'``.
    n_steps : int
        Number of time steps in each simulated path.
    n_paths : int
        Number of independent paths to generate.
    random_state : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray of shape (n_paths, n_steps)
        Simulated return paths, where each row is one path.

    Raises
    ------
    ValueError
        If ``params['distribution']`` is not recognised.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.normal(0, 0.01, 500)
    >>> params = fit_normal(returns)
    >>> paths = simulate_parametric(params, n_steps=252, n_paths=100)
    >>> paths.shape
    (100, 252)
    """
    rng = np.random.default_rng(random_state)
    dist = params.get("distribution")

    if dist == "normal":
        return rng.normal(loc=params["mu"], scale=params["sigma"], size=(n_paths, n_steps))

    if dist == "student_t":
        # scipy's t.rvs uses loc/scale; replicate with numpy for speed
        std_t = rng.standard_t(df=params["df"], size=(n_paths, n_steps))
        return params["mu"] + params["sigma"] * std_t

    raise ValueError(
        f"Unknown distribution '{dist}'. Expected 'normal' or 'student_t'."
    )
