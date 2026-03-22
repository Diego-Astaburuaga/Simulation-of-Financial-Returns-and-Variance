"""src.models – return and variance simulation models.

Implemented
-----------
bootstrap   – non-parametric resampling simulation
parametric  – Normal and Student-t distribution simulation

Placeholders (not yet implemented)
-----------------------------------
garch       – GARCH / HAR volatility models
gbm         – Geometric Brownian Motion
heston      – Heston stochastic volatility model
ml_models   – VAE and GAN generative models
"""

from src.models.bootstrap import simulate_bootstrap
from src.models.parametric import fit_normal, fit_student_t, simulate_parametric

__all__ = [
    "simulate_bootstrap",
    "fit_normal",
    "fit_student_t",
    "simulate_parametric",
]
