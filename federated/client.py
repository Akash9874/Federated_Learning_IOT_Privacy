"""
Federated Learning Client
=========================
Simulates an IoT device participating in federated learning.
Includes battery simulation, latency, and local training capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import time
from typing import Dict, Tuple, Optional, OrderedDict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.model import HARModel, create_model
from optimization.energy import EnergyManager
from optimization.compression import ModelCompressor
from privacy.differential_privacy import DifferentialPrivacy


class IoTClient:
    """
    Simulates an IoT device (client) in federated learning.
    
    Features:
    - Local model training
    - Battery level simulation
    - Network latency simulation
    - Energy-aware training (dynamic epochs)
    - Model update compression
    - Differential privacy support
    """
    
    def __init__(
        self,
        client_id: int,
        train_dataset,
        test_dataset,
        device: torch.device,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        initial_battery: float = 100.0,
        latency_range: Tuple[float, float] = (0.1, 2.0),
        enable_dp: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        enable_compression: bool = False,
        compression_ratio: float = 0.5
    ):
        """
        Initialize an IoT client.
        
        Args:
            client_id: Unique identifier for this client
            train_dataset: Training dataset for this client
            test_dataset: Test dataset for this client
            device: PyTorch device
            local_epochs: Number of local training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            initial_battery: Initial battery level (0-100)
            latency_range: (min, max) network latency in seconds
            enable_dp: Enable differential privacy
            dp_epsilon: Privacy budget epsilon
            dp_delta: Privacy budget delta
            enable_compression: Enable model compression
            compression_ratio: Compression ratio (0-1)
        """
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        ) if train_dataset else None
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        ) if test_dataset else None
        
        self.train_size = len(train_dataset) if train_dataset else 0
        self.test_size = len(test_dataset) if test_dataset else 0
        
        # Initialize local model
        self.model = create_model(device)
        
        # Energy management
        self.energy_manager = EnergyManager(
            initial_battery=initial_battery,
            client_id=client_id
        )
        
        # Network simulation
        self.latency_range = latency_range
        self.current_latency = np.random.uniform(*latency_range)
        
        # Differential privacy
        self.enable_dp = enable_dp
        self.dp = DifferentialPrivacy(
            epsilon=dp_epsilon,
            delta=dp_delta,
            max_grad_norm=1.0
        ) if enable_dp else None
        
        # Compression
        self.enable_compression = enable_compression
        self.compressor = ModelCompressor(
            compression_ratio=compression_ratio
        ) if enable_compression else None
        
        # Training statistics
        self.training_history = []
        self.communication_cost = 0
        
    def set_model_weights(self, weights: OrderedDict) -> None:
        """Set the local model weights from global model."""
        self.model.set_weights(weights)
    
    def get_model_weights(self) -> OrderedDict:
        """Get the current local model weights."""
        return self.model.get_weights()
    
    def get_battery_level(self) -> float:
        """Get current battery level."""
        return self.energy_manager.get_battery_level()
    
    def get_latency(self) -> float:
        """Get current simulated network latency."""
        # Simulate varying latency
        self.current_latency = np.random.uniform(*self.latency_range)
        return self.current_latency
    
    def is_available(self, min_battery: float = 20.0) -> bool:
        """Check if client is available for training."""
        return self.energy_manager.can_train(min_battery)
    
    def train_local(
        self, 
        global_weights: OrderedDict,
        adaptive_epochs: bool = True
    ) -> Tuple[OrderedDict, Dict]:
        """
        Perform local training on client's data.
        
        Args:
            global_weights: Global model weights to start from
            adaptive_epochs: Whether to adjust epochs based on battery
            
        Returns:
            Tuple of (updated_weights, training_stats)
        """
        if self.train_loader is None:
            raise ValueError(f"Client {self.client_id} has no training data")
        
        # Set model to global weights
        self.set_model_weights(global_weights)
        
        # Determine number of epochs (energy-aware)
        if adaptive_epochs:
            epochs = self.energy_manager.get_adaptive_epochs(
                base_epochs=self.local_epochs
            )
        else:
            epochs = self.local_epochs
        
        # Setup training
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate,
            momentum=0.9
        )
        
        start_time = time.time()
        total_loss = 0
        total_samples = 0
        
        # Local training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping for differential privacy
                if self.enable_dp:
                    self.dp.clip_gradients(self.model)
                
                optimizer.step()
                
                epoch_loss += loss.item() * len(data)
                total_samples += len(data)
            
            total_loss += epoch_loss
        
        training_time = time.time() - start_time
        
        # Consume energy for training
        energy_consumed = self.energy_manager.consume_training_energy(epochs)
        
        # Get updated weights
        updated_weights = self.get_model_weights()
        
        # Apply differential privacy noise
        if self.enable_dp:
            updated_weights = self.dp.add_noise_to_weights(
                updated_weights, 
                self.train_size
            )
        
        # Compress weights if enabled
        original_size = self._calculate_model_size(updated_weights)
        if self.enable_compression:
            updated_weights, compressed_size = self.compressor.compress(updated_weights)
        else:
            compressed_size = original_size
        
        self.communication_cost += compressed_size
        
        # Simulate network latency
        latency = self.get_latency()
        time.sleep(min(latency, 0.1))  # Cap actual sleep for simulation
        
        # Compile statistics
        stats = {
            'client_id': self.client_id,
            'epochs_trained': epochs,
            'train_loss': total_loss / total_samples,
            'train_samples': self.train_size,
            'training_time': training_time,
            'battery_level': self.get_battery_level(),
            'energy_consumed': energy_consumed,
            'communication_size': compressed_size,
            'compression_ratio': compressed_size / original_size if original_size > 0 else 1.0,
            'latency': latency
        }
        
        self.training_history.append(stats)
        
        return updated_weights, stats
    
    def evaluate_local(self) -> Dict:
        """
        Evaluate the model on local test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.test_loader is None:
            return {'accuracy': 0, 'loss': float('inf')}
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        return {
            'client_id': self.client_id,
            'accuracy': 100.0 * correct / total if total > 0 else 0,
            'loss': total_loss / total if total > 0 else float('inf'),
            'test_samples': total
        }
    
    def _calculate_model_size(self, weights: OrderedDict) -> int:
        """Calculate the size of model weights in bytes."""
        total_size = 0
        for key, value in weights.items():
            total_size += value.numel() * value.element_size()
        return total_size
    
    def reset_battery(self, level: float = 100.0) -> None:
        """Reset battery level (simulates charging)."""
        self.energy_manager.battery_level = level
    
    def get_statistics(self) -> Dict:
        """Get comprehensive client statistics."""
        return {
            'client_id': self.client_id,
            'train_samples': self.train_size,
            'test_samples': self.test_size,
            'battery_level': self.get_battery_level(),
            'total_communication_cost': self.communication_cost,
            'training_rounds': len(self.training_history),
            'dp_enabled': self.enable_dp,
            'compression_enabled': self.enable_compression
        }


class ClientManager:
    """
    Manages multiple IoT clients in federated learning.
    """
    
    def __init__(self):
        self.clients: Dict[int, IoTClient] = {}
    
    def add_client(self, client: IoTClient) -> None:
        """Add a client to the manager."""
        self.clients[client.client_id] = client
    
    def get_client(self, client_id: int) -> Optional[IoTClient]:
        """Get a client by ID."""
        return self.clients.get(client_id)
    
    def get_all_clients(self) -> list:
        """Get list of all clients."""
        return list(self.clients.values())
    
    def get_available_clients(self, min_battery: float = 20.0) -> list:
        """Get list of available clients."""
        return [c for c in self.clients.values() if c.is_available(min_battery)]
    
    def get_total_samples(self) -> int:
        """Get total number of training samples across all clients."""
        return sum(c.train_size for c in self.clients.values())
    
    def reset_all_batteries(self) -> None:
        """Reset all client batteries."""
        for client in self.clients.values():
            client.reset_battery()


if __name__ == "__main__":
    # Test client creation
    from data.har_loader import HARDataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    har_loader = HARDataLoader()
    client_data = har_loader.get_client_data()
    
    # Create a test client
    test_client_id = list(client_data.keys())[0]
    test_data = client_data[test_client_id]
    
    client = IoTClient(
        client_id=test_client_id,
        train_dataset=test_data['train'],
        test_dataset=test_data['test'],
        device=device,
        enable_dp=True,
        enable_compression=True
    )
    
    print(f"Client {client.client_id} created")
    print(f"Battery level: {client.get_battery_level():.1f}%")
    print(f"Train samples: {client.train_size}")
