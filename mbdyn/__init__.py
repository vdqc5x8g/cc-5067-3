"""mbdyn: multiband dynamics processor (split -> detect/gain -> recombine)."""

from .bands import (
    BandProcessing,
    band_ranges,
    configs_for_topology,
    match_band_intervals,
)
from .crossover import AllpassCascade, MultibandCrossover
from .dynamics import BandConfig, BandDynamics, static_gain_db
from .engine import MultibandEngine

__version__ = "1.0.0"
__all__ = [
    "AllpassCascade",
    "BandConfig",
    "BandDynamics",
    "BandProcessing",
    "MultibandCrossover",
    "MultibandEngine",
    "band_ranges",
    "configs_for_topology",
    "match_band_intervals",
    "static_gain_db",
]
