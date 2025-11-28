"""Gait analysis module for computing gait metrics and baseline modeling."""

from .gait_features import GaitAnalyzer
from .baseline import BaselineModel

__all__ = ['GaitAnalyzer', 'BaselineModel']