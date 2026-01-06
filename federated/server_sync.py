"""
Synchronous Federated Learning Server
=====================================
Implements FedAvg algorithm where the server waits for all
selected clients before aggregating updates.
"""

import torch
import numpy as np
import copy
import time
from typing import Dict, List, Tuple, Optional, OrderedDict
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.model import HARModel, create_model, ModelManager
from federated.client import IoTClient, ClientManager
from optimization.client_selection import AdaptiveClientSelector
from evaluation.metrics import MetricsTracker
from torch.utils.data import DataLoader


class SyncFederatedServer:
    """
    Synchronous Federated Learning Server implementing FedAvg.
    
    The server:
    1. Selects a subset of clients each round
    2. Distributes the global model
    3. Waits for ALL selected clients to complete training
    4. Aggregates updates using weighted averaging
    """
    
    def __init__(
        self,
        device: torch.device,
        num_rounds: int = 50,
        clients_per_round: int = 10,
        min_clients: int = 5,
        learning_rate: float = 1.0,
        adaptive_selection: bool = True,
        min_battery: float = 20.0,
        max_latency: float = 2.0
    ):
        """
        Initialize the synchronous FL server.
        
        Args:
            device: PyTorch device
            num_rounds: Number of federated learning rounds
            clients_per_round: Target number of clients per round
            min_clients: Minimum clients needed to proceed with a round
            learning_rate: Server learning rate for aggregation
            adaptive_selection: Use adaptive client selection
            min_battery: Minimum battery for client selection
            max_latency: Maximum latency for client selection
        """
        self.device = device
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.min_clients = min_clients
        self.learning_rate = learning_rate
        
        # Initialize global model
        self.global_model = create_model(device)
        self.global_weights = self.global_model.get_weights()
        
        # Client management
        self.client_manager = ClientManager()
        
        # Client selection
        self.adaptive_selection = adaptive_selection
        self.client_selector = AdaptiveClientSelector(
            min_battery=min_battery,
            max_latency=max_latency
        )
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        
        # Training history
        self.round_history = []
        
        # Global test dataset for evaluation
        self.test_loader = None
        
    def register_clients(self, clients: List[IoTClient]) -> None:
        """Register clients with the server."""
        for client in clients:
            self.client_manager.add_client(client)
        print(f"Registered {len(clients)} clients with the server")
    
    def set_test_dataset(self, test_dataset, batch_size: int = 64) -> None:
        """Set global test dataset for evaluation."""
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def select_clients(self, round_num: int) -> List[IoTClient]:
        """
        Select clients for the current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected clients
        """
        all_clients = self.client_manager.get_all_clients()
        
        if self.adaptive_selection:
            selected = self.client_selector.select_clients(
                clients=all_clients,
                num_to_select=self.clients_per_round,
                round_num=round_num
            )
        else:
            # Random selection from available clients
            available = self.client_manager.get_available_clients()
            num_select = min(self.clients_per_round, len(available))
            selected = np.random.choice(
                available, 
                size=num_select, 
                replace=False
            ).tolist()
        
        return selected
    
    def aggregate_updates(
        self, 
        client_weights: List[OrderedDict],
        sample_counts: List[int]
    ) -> OrderedDict:
        """
        Aggregate client model updates using FedAvg.
        
        Args:
            client_weights: List of client model weights
            sample_counts: Number of samples from each client
            
        Returns:
            Aggregated global model weights
        """
        return ModelManager.average_weights(client_weights, sample_counts)
    
    def train_round(self, round_num: int) -> Dict:
        """
        Execute one round of federated learning.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary with round statistics
        """
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        if len(selected_clients) < self.min_clients:
            print(f"Round {round_num}: Not enough clients available ({len(selected_clients)} < {self.min_clients})")
            return {'status': 'skipped', 'reason': 'insufficient_clients'}
        
        # Collect updates from all selected clients (synchronous)
        client_weights = []
        sample_counts = []
        client_stats = []
        
        for client in selected_clients:
            try:
                # Client performs local training
                weights, stats = client.train_local(
                    global_weights=self.global_weights,
                    adaptive_epochs=True
                )
                client_weights.append(weights)
                sample_counts.append(stats['train_samples'])
                client_stats.append(stats)
            except Exception as e:
                print(f"Client {client.client_id} failed: {e}")
                continue
        
        if len(client_weights) == 0:
            return {'status': 'failed', 'reason': 'no_successful_clients'}
        
        # Aggregate updates (FedAvg)
        self.global_weights = self.aggregate_updates(client_weights, sample_counts)
        self.global_model.set_weights(self.global_weights)
        
        round_time = time.time() - round_start_time
        
        # Evaluate global model
        eval_results = self.evaluate_global_model()
        
        # Compile round statistics
        round_stats = {
            'round': round_num,
            'status': 'completed',
            'num_clients': len(client_weights),
            'total_samples': sum(sample_counts),
            'round_time': round_time,
            'avg_client_loss': np.mean([s['train_loss'] for s in client_stats]),
            'avg_battery': np.mean([s['battery_level'] for s in client_stats]),
            'total_communication': sum([s['communication_size'] for s in client_stats]),
            'global_accuracy': eval_results['accuracy'],
            'global_loss': eval_results['loss']
        }
        
        self.round_history.append(round_stats)
        self.metrics.log_round(round_stats)
        
        return round_stats
    
    def evaluate_global_model(self) -> Dict:
        """
        Evaluate the global model on global test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.global_model.eval()
        self.global_model.set_weights(self.global_weights)
        
        # Use global test loader if available
        if self.test_loader is not None:
            total_correct = 0
            total_samples = 0
            total_loss = 0
            criterion = torch.nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item() * len(data)
                    pred = output.argmax(dim=1)
                    total_correct += pred.eq(target).sum().item()
                    total_samples += len(data)
            
            accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
            loss = total_loss / total_samples if total_samples > 0 else float('inf')
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'total_samples': total_samples
            }
        
        # Fallback: Evaluate on clients' test data
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        for client in self.client_manager.get_all_clients():
            if client.test_loader is None:
                continue
            
            client.set_model_weights(self.global_weights)
            results = client.evaluate_local()
            
            if results['test_samples'] > 0:
                total_correct += results['accuracy'] * results['test_samples'] / 100
                total_loss += results['loss'] * results['test_samples']
                total_samples += results['test_samples']
        
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'total_samples': total_samples
        }
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Execute the full federated learning process.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("SYNCHRONOUS FEDERATED LEARNING")
        print("="*60)
        
        training_start_time = time.time()
        
        # Training loop
        for round_num in tqdm(range(1, self.num_rounds + 1), desc="FL Rounds"):
            round_stats = self.train_round(round_num)
            
            if verbose and round_stats.get('status') == 'completed':
                if round_num % 5 == 0 or round_num == 1:
                    print(f"\nRound {round_num}: "
                          f"Accuracy={round_stats['global_accuracy']:.2f}%, "
                          f"Loss={round_stats['global_loss']:.4f}, "
                          f"Clients={round_stats['num_clients']}")
        
        total_time = time.time() - training_start_time
        
        # Final evaluation
        final_eval = self.evaluate_global_model()
        
        results = {
            'method': 'Synchronous FL (FedAvg)',
            'num_rounds': self.num_rounds,
            'total_time': total_time,
            'final_accuracy': final_eval['accuracy'],
            'final_loss': final_eval['loss'],
            'round_history': self.round_history,
            'total_communication': sum(r.get('total_communication', 0) for r in self.round_history)
        }
        
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final Accuracy: {final_eval['accuracy']:.2f}%")
        
        return results
    
    def get_global_model(self) -> HARModel:
        """Get the trained global model."""
        return self.global_model
    
    def save_model(self, path: str) -> None:
        """Save the global model to disk."""
        torch.save(self.global_weights, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a model from disk."""
        self.global_weights = torch.load(path)
        self.global_model.set_weights(self.global_weights)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from data.har_loader import HARDataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    har_loader = HARDataLoader()
    client_data = har_loader.get_client_data()
    
    # Create clients
    clients = []
    for client_id, data in client_data.items():
        if data['train'] is not None:
            client = IoTClient(
                client_id=client_id,
                train_dataset=data['train'],
                test_dataset=data['test'],
                device=device,
                local_epochs=3,
                enable_dp=False,
                enable_compression=False
            )
            clients.append(client)
    
    # Create and run server
    server = SyncFederatedServer(
        device=device,
        num_rounds=10,
        clients_per_round=5
    )
    server.register_clients(clients)
    results = server.train()
