"""Plotting utilities for financial return simulations."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_sample_paths(paths_dict, n_display=50, title="Simulated Return Paths", figsize=(12, 5)):
    """Plot a selection of simulated cumulative return paths.

    Parameters
    ----------
    paths_dict : dict of {str: ndarray}
        Mapping from label to a 2-D array of shape ``(n_paths, n_steps)``
        containing **cumulative** returns.
    n_display : int, optional
        Maximum number of paths to draw per series (default 50).
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size passed to ``plt.figure``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(paths_dict), figsize=figsize, sharey=True)
    if len(paths_dict) == 1:
        axes = [axes]

    for ax, (label, paths) in zip(axes, paths_dict.items()):
        n = min(n_display, paths.shape[0])
        for i in range(n):
            ax.plot(paths[i], alpha=0.3, linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative return")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_return_histogram(returns_dict, bins=60, title="Return Distribution", figsize=(10, 5)):
    """Plot overlapping histograms (and fitted Normal KDE) for return series.

    Parameters
    ----------
    returns_dict : dict of {str: array-like}
        Mapping from label to a 1-D return array.
    bins : int, optional
        Number of histogram bins (default 60).
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, returns in returns_dict.items():
        returns = np.asarray(returns).ravel()
        ax.hist(returns, bins=bins, density=True, alpha=0.4, label=label)
        x = np.linspace(returns.min(), returns.max(), 300)
        mu, sigma = np.mean(returns), np.std(returns, ddof=1)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=1.5)

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_volatility_clustering(returns, window=21, title="Volatility Clustering", figsize=(12, 4)):
    """Plot rolling standard deviation to visualise volatility clustering.

    Parameters
    ----------
    returns : array-like of shape (n,)
        1-D return series.
    window : int, optional
        Rolling window size in time steps (default 21 ≈ 1 trading month).
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    returns = pd.Series(np.asarray(returns).ravel())
    rolling_vol = returns.rolling(window).std()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(returns.values, linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("Return")
    axes[0].set_title(title)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")

    axes[1].plot(rolling_vol.values, linewidth=1.0, color="firebrick")
    axes[1].set_ylabel(f"Rolling std ({window}-step)")
    axes[1].set_xlabel("Time step")

    fig.tight_layout()
    return fig
