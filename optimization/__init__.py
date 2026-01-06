# Optimization module initialization
from .client_selection import AdaptiveClientSelector, PriorityClientSelector
from .energy import EnergyManager, EnergyAwareScheduler
from .compression import ModelCompressor, GradientAccumulator, calculate_communication_cost
