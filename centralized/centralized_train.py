"""
Centralized Training Baseline
=============================
Standard centralized training where all data is pooled together.
Used as a baseline to compare with federated learning approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Tuple
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.model import HARModel, create_model
from data.har_loader import HARDataLoader
from evaluation.metrics import MetricsTracker


class CentralizedTrainer:
    """
    Centralized training baseline.
    
    All data is pooled together and trained on a single model.
    This represents the upper bound of performance (no data distribution issues).
    """
    
    def __init__(
        self,
        device: torch.device,
        num_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize the centralized trainer.
        
        Args:
            device: PyTorch device
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        self.model = create_model(device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        self.training_history = []
        
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load centralized training data."""
        har_loader = HARDataLoader()
        train_loader, test_loader = har_loader.get_data_loaders(
            batch_size=self.batch_size
        )
        return train_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(data)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * len(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Execute the full centralized training process.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("CENTRALIZED LEARNING BASELINE")
        print("="*60)
        
        # Load data
        train_loader, test_loader = self.load_data()
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        training_start_time = time.time()
        best_accuracy = 0
        best_weights = None
        
        # Training loop
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="Centralized Training"):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            test_loss, test_acc = self.evaluate(test_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_weights = self.model.state_dict().copy()
            
            # Log metrics
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"\nEpoch {epoch}: "
                      f"Train Acc={train_acc:.2f}%, "
                      f"Test Acc={test_acc:.2f}%, "
                      f"Loss={test_loss:.4f}")
        
        total_time = time.time() - training_start_time
        
        # Final evaluation with best model
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        final_loss, final_accuracy = self.evaluate(test_loader)
        
        results = {
            'method': 'Centralized Learning',
            'num_epochs': self.num_epochs,
            'total_time': total_time,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'best_accuracy': best_accuracy,
            'training_history': self.training_history,
            'round_history': [{'global_accuracy': h['test_accuracy'], 'global_loss': h['test_loss']} 
                            for h in self.training_history],
            'train_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset)
        }
        
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final Accuracy: {final_accuracy:.2f}%")
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        
        return results
    
    def get_model(self) -> HARModel:
        """Get the trained model."""
        return self.model
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def run_centralized_experiment(
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> Dict:
    """
    Run centralized training experiment.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary with experiment results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = CentralizedTrainer(
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    results = trainer.train()
    return results


if __name__ == "__main__":
    results = run_centralized_experiment(num_epochs=20)
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING RESULTS")
    print("="*60)
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Training Time: {results['total_time']:.2f}s")
