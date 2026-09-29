"""Ensemble strategies for model combination."""

from .base import _Base
from .soft_voting import SoftVoting
from .stacking import StackingEnsemble
from .weighted_voting import WeightedVoting

__all__ = ["_Base", "SoftVoting", "WeightedVoting", "StackingEnsemble"]
