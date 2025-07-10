"""
DSPO (Diffusion Statistical Preference Optimization) package.

This package implements preference-based fine-tuning for diffusion models using
human feedback and multi-head reward modeling.

Modules:
    datasets: Data loading and preprocessing for preference pairs
    reward: Multi-head reward model with OpenCLIP backbone
    tuner: DSPO fine-tuning implementation with DoRA
"""

from .datasets import PairPreferenceDataset
from .reward import MultiHeadReward
from .tuner import DSPOFineTuner

__version__ = "0.1.0"
__all__ = ["PairPreferenceDataset", "MultiHeadReward", "DSPOFineTuner"]
