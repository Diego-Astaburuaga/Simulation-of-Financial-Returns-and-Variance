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
├── src/                             # All importable Python source code
│   ├── __init__.py
│   │
│   ├── models/                      # One file per model family
│   │   ├── __init__.py              # Re-exports public API
│   │   ├── bootstrap.py             # Bootstrap (non-parametric) simulation  ✓ implemented
│   │   ├── parametric.py            # Normal & Student-t simulation           ✓ implemented
│   │   ├── garch.py                 # GARCH / HAR volatility models           ☐ placeholder
│   │   ├── gbm.py                   # Geometric Brownian Motion               ☐ placeholder
│   │   ├── heston.py                # Heston stochastic volatility            ☐ placeholder
│   │   └── ml_models.py             # VAE / GAN generative models             ☐ placeholder
│   │
│   ├── simulation/                  # Engines that orchestrate model calls
│   │   ├── __init__.py
│   │   └── monte_carlo.py           # Generic Monte Carlo engine              ✓ implemented
│   │
│   └── utils/                       # Shared helpers (no domain logic)
│       ├── __init__.py
│       ├── statistics.py            # Summary statistics helpers               ✓ implemented
│       └── plotting.py              # Visualisation utilities                  ✓ implemented
│
├── notebooks/                       # Exploratory / demo notebooks
│   └── 01_bootstrap_vs_parametric.ipynb
│
├── tests/                           # pytest test suite (mirrors src/ layout)
│   ├── __init__.py
│   ├── test_bootstrap.py
│   ├── test_parametric.py
│   ├── test_monte_carlo.py
│   └── test_statistics.py
│
├── results/                         # Output figures / CSVs (git-ignored)
├── reports/theory/                  # Theoretical notes / PDFs
├── requirements.txt
└── README.md
```

### Directory conventions

| Directory | Rule |
|---|---|
| `src/models/` | One `.py` file per model family. Each file exposes `fit_*` and `simulate_*` functions. |
| `src/simulation/` | Engines that call models; no fitting logic here. |
| `src/utils/` | Pure helpers: statistics and plotting only. No model logic. |
| `notebooks/` | Numbered `NN_<topic>.ipynb`. Keep cells short; import from `src/` rather than re-implementing logic. |
| `tests/` | Mirror the `src/` layout: `test_<module>.py` for each source file. Use `pytest`. |
| `data/` | Raw and processed data files. Never committed output (figures, predictions). |
| `results/` | Generated artefacts (figures, CSVs). Git-ignored. |
| `reports/theory/` | PDF / Markdown theoretical notes. |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test suite
pytest tests/

# 3. Launch the notebook
jupyter notebook notebooks/01_bootstrap_vs_parametric.ipynb
```

---

## Implemented Models

| Model | Module | Key function | Status |
|---|---|---|---|
| Bootstrap (non-parametric) | `src/models/bootstrap.py` | `simulate_bootstrap` | ✓ |
| Normal distribution | `src/models/parametric.py` | `fit_normal`, `simulate_parametric` | ✓ |
| Student-t distribution | `src/models/parametric.py` | `fit_student_t`, `simulate_parametric` | ✓ |
| Monte Carlo engine | `src/simulation/monte_carlo.py` | `run_monte_carlo` | ✓ |
| GARCH / HAR | `src/models/garch.py` | `fit_garch`, `simulate_garch`, `fit_har` | ☐ placeholder |
| GBM | `src/models/gbm.py` | `fit_gbm`, `simulate_gbm` | ☐ placeholder |
| Heston | `src/models/heston.py` | `fit_heston`, `simulate_heston` | ☐ placeholder |
| VAE / GAN | `src/models/ml_models.py` | `train_vae`, `train_gan` | ☐ placeholder |

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
- [ ] GARCH / HAR volatility models  → `src/models/garch.py`
- [ ] Geometric Brownian Motion (GBM)  → `src/models/gbm.py`
- [ ] Heston stochastic volatility model  → `src/models/heston.py`
- [ ] ML generative models (VAE, GAN)  → `src/models/ml_models.py`

---

## Dependencies

- Python ≥ 3.10
- NumPy, Pandas, SciPy, Matplotlib, Jupyter
