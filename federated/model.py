"""
Neural Network Model for Human Activity Recognition
====================================================
A multi-layer perceptron designed for the HAR dataset.
Used across centralized, synchronous FL, and asynchronous FL experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, OrderedDict
import copy


class HARModel(nn.Module):
    """
    Multi-layer Perceptron for Human Activity Recognition.
    
    Architecture:
    - Input: 561 features
    - Hidden Layer 1: 256 neurons + BatchNorm + ReLU + Dropout
    - Hidden Layer 2: 128 neurons + BatchNorm + ReLU + Dropout
    - Hidden Layer 3: 64 neurons + BatchNorm + ReLU + Dropout
    - Output: 6 classes (activities)
    """
    
    def __init__(
        self, 
        input_dim: int = 561, 
        num_classes: int = 6,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the HAR model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(HARModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Layer 1: 561 -> 256
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 -> 64
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer: 64 -> 6
        self.fc_out = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 561)
            
        Returns:
            Output tensor of shape (batch_size, 6)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc_out(x)
        
        return x
    
    def get_weights(self) -> OrderedDict:
        """
        Get model weights as an OrderedDict.
        
        Returns:
            OrderedDict of model parameters
        """
        return copy.deepcopy(self.state_dict())
    
    def set_weights(self, weights: OrderedDict) -> None:
        """
        Set model weights from an OrderedDict.
        
        Args:
            weights: OrderedDict of model parameters
        """
        self.load_state_dict(weights)
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get model gradients.
        
        Returns:
            Dictionary mapping parameter names to gradients
        """
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelManager:
    """
    Utility class for model operations in federated learning.
    """
    
    @staticmethod
    def average_weights(
        weights_list: list, 
        sample_counts: list = None
    ) -> OrderedDict:
        """
        Average model weights (FedAvg algorithm).
        
        Args:
            weights_list: List of OrderedDicts containing model weights
            sample_counts: Optional list of sample counts for weighted averaging
            
        Returns:
            Averaged weights as OrderedDict
        """
        if not weights_list:
            raise ValueError("weights_list cannot be empty")
        
        # Use uniform weights if sample counts not provided
        if sample_counts is None:
            sample_counts = [1] * len(weights_list)
        
        total_samples = sum(sample_counts)
        
        # Initialize averaged weights with zeros
        averaged_weights = copy.deepcopy(weights_list[0])
        for key in averaged_weights.keys():
            averaged_weights[key] = torch.zeros_like(averaged_weights[key], dtype=torch.float32)
        
        # Weighted sum of all model weights
        for weights, count in zip(weights_list, sample_counts):
            weight_factor = count / total_samples
            for key in averaged_weights.keys():
                averaged_weights[key] += weights[key].float() * weight_factor
        
        return averaged_weights
    
    @staticmethod
    def compute_weight_update(
        old_weights: OrderedDict, 
        new_weights: OrderedDict
    ) -> OrderedDict:
        """
        Compute the difference between new and old weights.
        
        Args:
            old_weights: Original model weights
            new_weights: Updated model weights
            
        Returns:
            Weight updates (delta)
        """
        update = OrderedDict()
        for key in old_weights.keys():
            update[key] = new_weights[key] - old_weights[key]
        return update
    
    @staticmethod
    def apply_weight_update(
        weights: OrderedDict, 
        update: OrderedDict, 
        learning_rate: float = 1.0
    ) -> OrderedDict:
        """
        Apply a weight update to existing weights.
        
        Args:
            weights: Current model weights
            update: Weight update to apply
            learning_rate: Scaling factor for the update
            
        Returns:
            Updated weights
        """
        new_weights = copy.deepcopy(weights)
        for key in weights.keys():
            new_weights[key] = weights[key] + learning_rate * update[key]
        return new_weights


def create_model(device: torch.device = None) -> HARModel:
    """
    Factory function to create and initialize a HAR model.
    
    Args:
        device: Device to place the model on
        
    Returns:
        Initialized HARModel
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HARModel()
    model = model.to(device)
    
    # Initialize weights using Xavier initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 561).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test weight operations
    weights = model.get_weights()
    print(f"Number of weight tensors: {len(weights)}")
