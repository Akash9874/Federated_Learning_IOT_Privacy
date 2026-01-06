"""
Energy-Aware Training (NOVELTY MODULE)
======================================
Simulates energy consumption on IoT devices and adapts training
parameters based on available battery level.
"""

import numpy as np
from typing import Dict, Tuple


class EnergyManager:
    """
    Manages energy simulation for IoT devices.
    
    NOVELTY: Dynamically adjusts local training epochs based on
    battery level to extend device lifetime while maintaining
    learning effectiveness.
    
    Energy Model:
    - Training consumes energy per epoch
    - Communication consumes energy per transmission
    - Idle consumption is negligible
    """
    
    def __init__(
        self,
        initial_battery: float = 100.0,
        energy_per_epoch: float = 2.0,
        energy_per_communication: float = 1.0,
        idle_power: float = 0.01,
        client_id: int = 0
    ):
        """
        Initialize the energy manager.
        
        Args:
            initial_battery: Initial battery level (0-100)
            energy_per_epoch: Energy consumed per training epoch (%)
            energy_per_communication: Energy for one model transmission (%)
            idle_power: Energy consumed per second when idle (%)
            client_id: Client identifier for logging
        """
        self.battery_level = initial_battery
        self.initial_battery = initial_battery
        self.energy_per_epoch = energy_per_epoch
        self.energy_per_communication = energy_per_communication
        self.idle_power = idle_power
        self.client_id = client_id
        
        # Energy tracking
        self.total_energy_consumed = 0.0
        self.training_energy = 0.0
        self.communication_energy = 0.0
        self.energy_history = []
        
    def get_battery_level(self) -> float:
        """Get current battery level."""
        return max(0, self.battery_level)
    
    def can_train(self, min_battery: float = 20.0) -> bool:
        """Check if device has enough battery to train."""
        return self.battery_level >= min_battery
    
    def get_adaptive_epochs(
        self,
        base_epochs: int = 5,
        min_epochs: int = 1,
        battery_thresholds: Tuple[float, float, float] = (80.0, 50.0, 30.0)
    ) -> int:
        """
        Determine number of local epochs based on battery level.
        
        NOVELTY: Energy-aware training - fewer epochs when battery is low.
        
        Battery Level -> Epochs:
        - > 80%: base_epochs (full training)
        - 50-80%: base_epochs - 1
        - 30-50%: base_epochs - 2
        - < 30%: min_epochs (minimal training)
        
        Args:
            base_epochs: Maximum epochs when battery is high
            min_epochs: Minimum epochs even when battery is low
            battery_thresholds: (high, medium, low) thresholds
            
        Returns:
            Number of epochs to train
        """
        high, medium, low = battery_thresholds
        
        if self.battery_level >= high:
            return base_epochs
        elif self.battery_level >= medium:
            return max(min_epochs, base_epochs - 1)
        elif self.battery_level >= low:
            return max(min_epochs, base_epochs - 2)
        else:
            return min_epochs
    
    def consume_training_energy(self, epochs: int) -> float:
        """
        Consume energy for training.
        
        Args:
            epochs: Number of epochs trained
            
        Returns:
            Energy consumed
        """
        energy = self.energy_per_epoch * epochs
        self.battery_level = max(0, self.battery_level - energy)
        self.training_energy += energy
        self.total_energy_consumed += energy
        
        self.energy_history.append({
            'type': 'training',
            'epochs': epochs,
            'energy': energy,
            'battery_after': self.battery_level
        })
        
        return energy
    
    def consume_communication_energy(self, model_size_mb: float = 1.0) -> float:
        """
        Consume energy for communication.
        
        Args:
            model_size_mb: Size of model being transmitted in MB
            
        Returns:
            Energy consumed
        """
        # Energy scales with model size
        energy = self.energy_per_communication * model_size_mb
        self.battery_level = max(0, self.battery_level - energy)
        self.communication_energy += energy
        self.total_energy_consumed += energy
        
        self.energy_history.append({
            'type': 'communication',
            'size_mb': model_size_mb,
            'energy': energy,
            'battery_after': self.battery_level
        })
        
        return energy
    
    def simulate_idle(self, seconds: float) -> float:
        """
        Simulate idle power consumption.
        
        Args:
            seconds: Duration of idle time
            
        Returns:
            Energy consumed
        """
        energy = self.idle_power * seconds
        self.battery_level = max(0, self.battery_level - energy)
        self.total_energy_consumed += energy
        return energy
    
    def recharge(self, amount: float = None) -> None:
        """
        Recharge the battery.
        
        Args:
            amount: Amount to recharge (None for full recharge)
        """
        if amount is None:
            self.battery_level = self.initial_battery
        else:
            self.battery_level = min(100.0, self.battery_level + amount)
    
    def get_energy_statistics(self) -> Dict:
        """Get comprehensive energy statistics."""
        return {
            'client_id': self.client_id,
            'current_battery': self.battery_level,
            'total_consumed': self.total_energy_consumed,
            'training_energy': self.training_energy,
            'communication_energy': self.communication_energy,
            'num_training_sessions': sum(
                1 for e in self.energy_history if e['type'] == 'training'
            ),
            'num_communications': sum(
                1 for e in self.energy_history if e['type'] == 'communication'
            )
        }
    
    def estimate_remaining_rounds(
        self,
        avg_epochs: int = 3,
        min_battery: float = 20.0
    ) -> int:
        """
        Estimate how many more rounds the device can participate in.
        
        Args:
            avg_epochs: Average epochs per round
            min_battery: Minimum battery threshold
            
        Returns:
            Estimated remaining rounds
        """
        available_energy = self.battery_level - min_battery
        if available_energy <= 0:
            return 0
        
        energy_per_round = (
            self.energy_per_epoch * avg_epochs +
            self.energy_per_communication * 2  # Send and receive
        )
        
        return int(available_energy / energy_per_round)


class EnergyAwareScheduler:
    """
    Schedules training across clients considering energy constraints.
    """
    
    def __init__(self, energy_budget_per_round: float = 50.0):
        """
        Initialize the energy-aware scheduler.
        
        Args:
            energy_budget_per_round: Total energy budget for all clients per round
        """
        self.energy_budget_per_round = energy_budget_per_round
        self.round_energy_usage = []
    
    def allocate_epochs(
        self,
        clients: list,
        base_epochs: int = 5
    ) -> Dict[int, int]:
        """
        Allocate epochs to clients within energy budget.
        
        Args:
            clients: List of IoTClient instances
            base_epochs: Base epochs to allocate
            
        Returns:
            Dictionary mapping client_id to epochs
        """
        allocations = {}
        remaining_budget = self.energy_budget_per_round
        
        # Sort clients by battery level (prioritize low-battery clients)
        sorted_clients = sorted(
            clients,
            key=lambda c: c.energy_manager.battery_level
        )
        
        for client in sorted_clients:
            em = client.energy_manager
            
            # Calculate adaptive epochs
            epochs = em.get_adaptive_epochs(base_epochs)
            
            # Check energy budget
            energy_needed = em.energy_per_epoch * epochs + em.energy_per_communication
            
            if energy_needed <= remaining_budget:
                allocations[client.client_id] = epochs
                remaining_budget -= energy_needed
            elif remaining_budget > em.energy_per_epoch + em.energy_per_communication:
                # Allocate minimum epochs
                allocations[client.client_id] = 1
                remaining_budget -= em.energy_per_epoch + em.energy_per_communication
        
        return allocations
    
    def get_round_efficiency(self) -> float:
        """Calculate energy efficiency over all rounds."""
        if not self.round_energy_usage:
            return 0.0
        return np.mean(self.round_energy_usage)


if __name__ == "__main__":
    # Test energy management
    em = EnergyManager(initial_battery=100.0)
    
    print(f"Initial battery: {em.get_battery_level():.1f}%")
    print(f"Can train: {em.can_train()}")
    
    # Simulate training rounds
    for i in range(10):
        epochs = em.get_adaptive_epochs(base_epochs=5)
        em.consume_training_energy(epochs)
        em.consume_communication_energy(1.0)
        print(f"Round {i+1}: {epochs} epochs, Battery: {em.get_battery_level():.1f}%")
    
    print("\nEnergy Statistics:")
    print(em.get_energy_statistics())
