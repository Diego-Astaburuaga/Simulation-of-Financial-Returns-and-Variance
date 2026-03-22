"""Summary statistics utilities for return series."""

import numpy as np
from scipy import stats


def compute_statistics(returns):
    """Compute descriptive statistics for a return series.

    Parameters
    ----------
    returns : array-like of shape (n,)
        Return series (e.g., daily log-returns).

    Returns
    -------
    dict
        Dictionary containing:

        - ``'mean'`` : float – arithmetic mean.
        - ``'std'`` : float – standard deviation (ddof=1).
        - ``'skewness'`` : float – Fisher skewness.
        - ``'kurtosis'`` : float – excess kurtosis (normal → 0).
        - ``'min'`` : float – minimum value.
        - ``'max'`` : float – maximum value.
        - ``'count'`` : int – number of observations.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.random.normal(0, 0.01, 500)
    >>> s = compute_statistics(r)
    >>> sorted(s.keys())
    ['count', 'kurtosis', 'max', 'mean', 'min', 'skewness', 'std']
    """
    returns = np.asarray(returns, dtype=float)
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns, ddof=1)),
        "skewness": float(stats.skew(returns)),
        "kurtosis": float(stats.kurtosis(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "count": len(returns),
    }


def compare_statistics(returns_dict):
    """Compute statistics for multiple return series and return a DataFrame.

    Parameters
    ----------
    returns_dict : dict of {str: array-like}
        Mapping from a label (e.g. ``'Historical'``) to a 1-D return array.

    Returns
    -------
    pandas.DataFrame
        Rows are statistic names, columns are series labels.

    Examples
    --------
    >>> import numpy as np
    >>> from src.utils.statistics import compare_statistics
    >>> d = {'A': np.random.normal(0, 0.01, 200),
    ...      'B': np.random.normal(0, 0.02, 200)}
    >>> df = compare_statistics(d)
    >>> list(df.columns)
    ['A', 'B']
    """
    import pandas as pd

    records = {label: compute_statistics(arr) for label, arr in returns_dict.items()}
    return pd.DataFrame(records)
