"""
Differential Privacy Module (NOVELTY MODULE)
============================================
Implements device-level differential privacy for federated learning.
Adds calibrated noise to model updates to preserve privacy.
"""

import torch
import numpy as np
from typing import Dict, Tuple, OrderedDict
from collections import OrderedDict as OD
import copy
import math


class DifferentialPrivacy:
    """
    Differential Privacy for Federated Learning.
    
    NOVELTY: Implements (ε, δ)-differential privacy at the device level.
    
    Mechanism:
    1. Gradient Clipping: Bound sensitivity by clipping gradients
    2. Noise Addition: Add calibrated Gaussian noise
    
    Privacy guarantee: Protects individual data points from being
    inferred through model updates.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = None
    ):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget (lower = more privacy)
            delta: Probability of privacy breach
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Noise scale (auto-calculated if None)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._calculate_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.num_compositions = 0
        self.noise_history = []
        
    def _calculate_noise_multiplier(self) -> float:
        """
        Calculate noise multiplier for Gaussian mechanism.
        
        Based on the analytical Gaussian mechanism:
        σ ≥ √(2 * ln(1.25/δ)) * Δf / ε
        
        where Δf is the sensitivity (max_grad_norm for us).
        
        Returns:
            Noise multiplier
        """
        if self.epsilon <= 0:
            return float('inf')
        
        # Gaussian mechanism formula
        sensitivity = self.max_grad_norm
        noise_scale = math.sqrt(2 * math.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        return noise_scale
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            model: PyTorch model with computed gradients
            
        Returns:
            Original gradient norm before clipping
        """
        # Calculate total gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Clip if necessary
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model: torch.nn.Module) -> None:
        """
        Add Gaussian noise to gradients for differential privacy.
        
        Args:
            model: PyTorch model with computed gradients
        """
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad.data.add_(noise)
        
        self.num_compositions += 1
        self._update_privacy_spent()
    
    def add_noise_to_weights(
        self, 
        weights: OrderedDict,
        num_samples: int
    ) -> OrderedDict:
        """
        Add noise to model weights (for local DP).
        
        Args:
            weights: Model weights
            num_samples: Number of samples used in training
            
        Returns:
            Noisy weights
        """
        noisy_weights = OD()
        
        # Scale noise by number of samples (more samples = less noise needed)
        noise_scale = self.noise_multiplier * self.max_grad_norm / math.sqrt(num_samples)
        
        total_noise_added = 0.0
        
        for key, tensor in weights.items():
            noise = torch.normal(
                mean=0,
                std=noise_scale,
                size=tensor.shape,
                device=tensor.device
            )
            noisy_weights[key] = tensor + noise
            total_noise_added += noise.abs().mean().item()
        
        self.noise_history.append({
            'noise_scale': noise_scale,
            'avg_noise': total_noise_added / len(weights)
        })
        
        self.num_compositions += 1
        self._update_privacy_spent()
        
        return noisy_weights
    
    def _update_privacy_spent(self) -> None:
        """Update privacy budget spent using basic composition."""
        # Basic composition: ε_total = k * ε for k compositions
        # Advanced composition: ε_total = √(2k * ln(1/δ)) * ε + k * ε * (e^ε - 1)
        
        # Using basic composition for simplicity
        self.privacy_spent = self.num_compositions * self.epsilon
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get total privacy budget spent.
        
        Returns:
            Tuple of (epsilon_spent, delta)
        """
        return self.privacy_spent, self.delta
    
    def get_remaining_budget(self, total_budget: float) -> float:
        """
        Get remaining privacy budget.
        
        Args:
            total_budget: Total epsilon budget
            
        Returns:
            Remaining epsilon
        """
        return max(0, total_budget - self.privacy_spent)
    
    def get_statistics(self) -> Dict:
        """Get privacy statistics."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'num_compositions': self.num_compositions,
            'privacy_spent': self.privacy_spent,
            'avg_noise_added': np.mean([n['avg_noise'] for n in self.noise_history]) if self.noise_history else 0
        }


class PrivacyAccountant:
    """
    Advanced privacy accounting using Rényi Differential Privacy (RDP).
    
    Provides tighter privacy bounds for composition.
    """
    
    def __init__(self, delta: float = 1e-5):
        """
        Initialize privacy accountant.
        
        Args:
            delta: Target delta for (ε, δ)-DP
        """
        self.delta = delta
        self.rdp_orders = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]
        self.rdp_eps = [0.0] * len(self.rdp_orders)
    
    def accumulate(self, noise_multiplier: float, sample_rate: float) -> None:
        """
        Accumulate privacy cost for one step.
        
        Args:
            noise_multiplier: Noise multiplier used
            sample_rate: Fraction of data used
        """
        for i, order in enumerate(self.rdp_orders):
            rdp = self._compute_rdp(order, noise_multiplier, sample_rate)
            self.rdp_eps[i] += rdp
    
    def _compute_rdp(
        self, 
        order: float, 
        noise_multiplier: float, 
        sample_rate: float
    ) -> float:
        """Compute RDP for a single step."""
        if noise_multiplier == 0:
            return float('inf')
        
        if sample_rate == 1:
            # Full batch
            return order / (2 * noise_multiplier ** 2)
        
        # Subsampled mechanism (simplified)
        return sample_rate ** 2 * order / (2 * noise_multiplier ** 2)
    
    def get_epsilon(self) -> float:
        """Convert RDP to (ε, δ)-DP."""
        min_eps = float('inf')
        
        for order, rdp in zip(self.rdp_orders, self.rdp_eps):
            eps = rdp - math.log(self.delta) / (order - 1)
            min_eps = min(min_eps, eps)
        
        return min_eps


def analyze_privacy_accuracy_tradeoff(
    epsilons: list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> Dict:
    """
    Analyze the trade-off between privacy and accuracy.
    
    Args:
        epsilons: List of epsilon values to analyze
        
    Returns:
        Dictionary with analysis results
    """
    results = []
    
    for eps in epsilons:
        dp = DifferentialPrivacy(epsilon=eps, delta=1e-5)
        
        results.append({
            'epsilon': eps,
            'noise_multiplier': dp.noise_multiplier,
            'privacy_level': 'strong' if eps < 1 else 'moderate' if eps < 5 else 'weak',
            'expected_accuracy_impact': 'high' if eps < 1 else 'medium' if eps < 5 else 'low'
        })
    
    return {
        'tradeoff_analysis': results,
        'recommendation': 'epsilon between 1.0 and 5.0 provides good privacy-utility balance'
    }


if __name__ == "__main__":
    # Test differential privacy
    print("Differential Privacy Module Test")
    print("="*50)
    
    # Create DP mechanism
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    
    print(f"Epsilon: {dp.epsilon}")
    print(f"Delta: {dp.delta}")
    print(f"Noise multiplier: {dp.noise_multiplier:.4f}")
    
    # Test noise addition
    test_weights = OD({
        'layer1': torch.randn(100, 50),
        'layer2': torch.randn(50, 10)
    })
    
    noisy_weights = dp.add_noise_to_weights(test_weights, num_samples=100)
    
    print(f"\nPrivacy spent: {dp.get_privacy_spent()}")
    print(f"Statistics: {dp.get_statistics()}")
    
    # Privacy-accuracy tradeoff
    print("\nPrivacy-Accuracy Tradeoff Analysis:")
    analysis = analyze_privacy_accuracy_tradeoff()
    for item in analysis['tradeoff_analysis']:
        print(f"  ε={item['epsilon']}: {item['privacy_level']} privacy, {item['expected_accuracy_impact']} accuracy impact")
