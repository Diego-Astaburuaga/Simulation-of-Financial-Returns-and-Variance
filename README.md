# Simulation of Financial Returns and Variance

A modular Python framework for simulating financial time series (returns and variance) using
data-driven, parametric, and stochastic methods.  The focus is on clarity, modularity, and
reproducibility — making it easy to extend with new models in the future.

---

## Project Structure

```
simulation-finance/
│
├── data/
│   └── example_returns.csv          # Synthetic daily log-returns
│
├── src/
│   ├── models/
│   │   ├── bootstrap.py             # Bootstrap (non-parametric) simulation
│   │   └── parametric.py            # Normal & Student-t simulation
│   │
│   ├── simulation/
│   │   └── monte_carlo.py           # Generic Monte Carlo engine
│   │
│   └── utils/
│       ├── statistics.py            # Summary statistics helpers
│       └── plotting.py              # Visualisation utilities
│
├── notebooks/
│   └── 01_bootstrap_vs_parametric.ipynb
│
├── results/                         # Output figures / CSVs (git-ignored)
├── reports/theory/                  # Theoretical notes
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebook
jupyter notebook notebooks/01_bootstrap_vs_parametric.ipynb
```

---

## Implemented Models

| Model | Module | Key function |
|---|---|---|
| Bootstrap (non-parametric) | `src/models/bootstrap.py` | `simulate_bootstrap` |
| Normal distribution | `src/models/parametric.py` | `fit_normal`, `simulate_parametric` |
| Student-t distribution | `src/models/parametric.py` | `fit_student_t`, `simulate_parametric` |
| Monte Carlo engine | `src/simulation/monte_carlo.py` | `run_monte_carlo` |

---

## Usage Example

```python
import numpy as np
import pandas as pd
from src.models.bootstrap import simulate_bootstrap
from src.models.parametric import fit_normal, fit_student_t, simulate_parametric
from src.simulation.monte_carlo import run_monte_carlo
from src.utils.statistics import compute_statistics
from src.utils.plotting import plot_sample_paths, plot_return_histogram

# Load data
df = pd.read_csv("data/example_returns.csv", parse_dates=["date"])
returns = df["return"].values

# Bootstrap simulation
boot_result = run_monte_carlo(
    simulate_bootstrap,
    n_steps=252,
    n_paths=500,
    returns=returns,
    random_state=42,
)

# Parametric simulation (Normal)
params_norm = fit_normal(returns)
norm_result = run_monte_carlo(
    simulate_parametric,
    n_steps=252,
    n_paths=500,
    params=params_norm,
    random_state=42,
)

# Compare statistics
print(compute_statistics(boot_result["returns"].ravel()))

# Visualise
fig = plot_sample_paths(
    {"Bootstrap": boot_result["cum_returns"], "Normal": norm_result["cum_returns"]},
    n_display=50,
)
fig.savefig("results/sample_paths.png", dpi=150)
```

---

## Roadmap

- [x] Bootstrap simulation
- [x] Parametric simulation (Normal, Student-t)
- [x] Monte Carlo engine
- [ ] GARCH / HAR volatility models
- [ ] Geometric Brownian Motion (GBM)
- [ ] Heston stochastic volatility model
- [ ] ML generative models (VAE, GAN)

---

## Dependencies

- Python ≥ 3.10
- NumPy, Pandas, SciPy, Matplotlib, Jupyter
