"""Imitation Learning Project: Modern implementation of behavioral cloning, GAIL, and AIRL.

This package provides a comprehensive implementation of imitation learning algorithms
for research and educational purposes. It includes behavioral cloning, generative
adversarial imitation learning (GAIL), and adversarial inverse reinforcement
learning (AIRL).

WARNING: This implementation is for research and educational purposes only.
It is NOT intended for production control of real-world systems.
"""

__version__ = "1.0.0"
__author__ = "RL Research Team"

from .algorithms import BehavioralCloning, GAIL, AIRL
from .models import PolicyNetwork, DiscriminatorNetwork, ValueNetwork
from .envs import ExpertDataCollector, EnvironmentWrapper
from .evaluation import ImitationEvaluator, MetricsTracker

__all__ = [
    "BehavioralCloning",
    "GAIL", 
    "AIRL",
    "PolicyNetwork",
    "DiscriminatorNetwork", 
    "ValueNetwork",
    "ExpertDataCollector",
    "EnvironmentWrapper",
    "ImitationEvaluator",
    "MetricsTracker",
]
