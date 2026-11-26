"""mbdyn: multiband dynamics processor (split -> detect/gain -> recombine)."""

from .crossover import AllpassCascade, MultibandCrossover
from .dynamics import BandConfig, BandDynamics, static_gain_db
from .engine import MultibandEngine

__version__ = "1.0.0"
__all__ = [
    "AllpassCascade",
    "BandConfig",
    "BandDynamics",
    "MultibandCrossover",
    "MultibandEngine",
    "static_gain_db",
]
