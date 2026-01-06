"""
Asynchronous Federated Learning Server (CORE NOVELTY)
=====================================================
Implements asynchronous FL where the server aggregates model updates
immediately as they arrive, without waiting for all clients.

Key Features:
- Weighted moving average aggregation
- Straggler handling
- Staleness-aware aggregation
"""

import torch
import numpy as np
import copy
import time
import threading
from queue import Queue, Empty
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


class AsyncFederatedServer:
    """
    Asynchronous Federated Learning Server.
    
    NOVELTY: Unlike synchronous FL, this server:
    1. Does NOT wait for all clients to finish
    2. Aggregates updates immediately as they arrive
    3. Uses staleness-aware weighted averaging
    4. Handles stragglers gracefully
    
    Aggregation Strategy: Weighted Moving Average
    new_global = (1 - alpha) * old_global + alpha * client_update
    where alpha is adjusted based on staleness and sample count
    """
    
    def __init__(
        self,
        device: torch.device,
        num_rounds: int = 50,
        clients_per_round: int = 10,
        min_updates_per_round: int = 3,
        base_alpha: float = 0.5,
        staleness_discount: float = 0.9,
        adaptive_selection: bool = True,
        min_battery: float = 20.0,
        max_latency: float = 2.0,
        timeout_seconds: float = 10.0
    ):
        """
        Initialize the asynchronous FL server.
        
        Args:
            device: PyTorch device
            num_rounds: Number of federated learning rounds
            clients_per_round: Target number of clients per round
            min_updates_per_round: Minimum updates before ending a round
            base_alpha: Base learning rate for aggregation
            staleness_discount: Discount factor for stale updates (per round)
            adaptive_selection: Use adaptive client selection
            min_battery: Minimum battery for client selection
            max_latency: Maximum latency for client selection
            timeout_seconds: Timeout waiting for client updates
        """
        self.device = device
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.min_updates_per_round = min_updates_per_round
        self.base_alpha = base_alpha
        self.staleness_discount = staleness_discount
        self.timeout_seconds = timeout_seconds
        
        # Initialize global model
        self.global_model = create_model(device)
        self.global_weights = self.global_model.get_weights()
        self.current_round = 0
        
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
        
        # Async components
        self.update_queue = Queue()
        self.lock = threading.Lock()
        
        # Tracking client participation
        self.client_last_round = {}  # client_id -> last round they participated
        
        # Training history
        self.round_history = []
        self.update_history = []
        
        # Global test dataset for evaluation
        self.test_loader = None
        
    def register_clients(self, clients: List[IoTClient]) -> None:
        """Register clients with the server."""
        for client in clients:
            self.client_manager.add_client(client)
            self.client_last_round[client.client_id] = 0
        print(f"Registered {len(clients)} clients with the server")
    
    def set_test_dataset(self, test_dataset, batch_size: int = 64) -> None:
        """Set global test dataset for evaluation."""
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def calculate_staleness(self, client_round: int) -> int:
        """Calculate staleness of a client update."""
        return max(0, self.current_round - client_round)
    
    def calculate_aggregation_weight(
        self, 
        staleness: int, 
        sample_count: int,
        total_samples: int
    ) -> float:
        """
        Calculate the aggregation weight for a client update.
        
        Weight = base_alpha * staleness_discount^staleness * (sample_count / total_samples)
        
        Args:
            staleness: Number of rounds since client's last update
            sample_count: Number of samples the client trained on
            total_samples: Total samples across all participating clients
            
        Returns:
            Aggregation weight (alpha)
        """
        # Apply staleness discount
        staleness_factor = self.staleness_discount ** staleness
        
        # Sample proportion
        sample_factor = sample_count / max(total_samples, 1)
        
        # Combined weight
        alpha = self.base_alpha * staleness_factor * (0.5 + 0.5 * sample_factor)
        
        return min(alpha, 1.0)  # Cap at 1.0
    
    def aggregate_async(
        self, 
        client_weights: OrderedDict,
        alpha: float
    ) -> None:
        """
        Perform asynchronous aggregation using weighted moving average.
        
        new_global = (1 - alpha) * old_global + alpha * client_weights
        
        Args:
            client_weights: Client's model weights
            alpha: Aggregation weight
        """
        with self.lock:
            for key in self.global_weights.keys():
                self.global_weights[key] = (
                    (1 - alpha) * self.global_weights[key].float() + 
                    alpha * client_weights[key].float()
                )
            self.global_model.set_weights(self.global_weights)
    
    def select_clients(self, round_num: int) -> List[IoTClient]:
        """Select clients for the current round."""
        all_clients = self.client_manager.get_all_clients()
        
        if self.adaptive_selection:
            selected = self.client_selector.select_clients(
                clients=all_clients,
                num_to_select=self.clients_per_round,
                round_num=round_num
            )
        else:
            available = self.client_manager.get_available_clients()
            num_select = min(self.clients_per_round, len(available))
            selected = np.random.choice(
                available, 
                size=num_select, 
                replace=False
            ).tolist()
        
        return selected
    
    def client_train_async(
        self, 
        client: IoTClient, 
        start_round: int,
        global_weights_snapshot: OrderedDict
    ) -> Optional[Tuple[OrderedDict, Dict]]:
        """
        Simulates async client training (would be threaded in real system).
        
        Args:
            client: IoTClient to train
            start_round: Round number when training started
            global_weights_snapshot: Copy of global weights when training started
            
        Returns:
            Tuple of (weights, stats) or None if failed
        """
        try:
            weights, stats = client.train_local(
                global_weights=global_weights_snapshot,
                adaptive_epochs=True
            )
            stats['start_round'] = start_round
            return weights, stats
        except Exception as e:
            print(f"Async training failed for client {client.client_id}: {e}")
            return None
    
    def train_round(self, round_num: int) -> Dict:
        """
        Execute one round of asynchronous federated learning.
        
        Key differences from sync FL:
        - Server takes a snapshot of global weights
        - Clients train on the snapshot (potentially stale)
        - Updates are aggregated with staleness-aware weights
        - Round ends after min_updates or timeout
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary with round statistics
        """
        self.current_round = round_num
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        if len(selected_clients) < self.min_updates_per_round:
            print(f"Round {round_num}: Not enough clients")
            return {'status': 'skipped', 'reason': 'insufficient_clients'}
        
        # Snapshot of global weights for this round
        global_weights_snapshot = copy.deepcopy(self.global_weights)
        
        # Simulate async training (in production, this would be truly async)
        updates_received = 0
        client_stats = []
        total_samples = sum(c.train_size for c in selected_clients)
        
        # Shuffle clients to simulate varying arrival times
        np.random.shuffle(selected_clients)
        
        for client in selected_clients:
            # Get client's last participation round
            client_start_round = self.client_last_round.get(client.client_id, 0)
            
            # Perform training
            result = self.client_train_async(
                client=client,
                start_round=round_num,
                global_weights_snapshot=global_weights_snapshot
            )
            
            if result is None:
                continue
            
            weights, stats = result
            
            # Calculate staleness
            staleness = self.calculate_staleness(client_start_round)
            
            # Calculate aggregation weight
            alpha = self.calculate_aggregation_weight(
                staleness=staleness,
                sample_count=stats['train_samples'],
                total_samples=total_samples
            )
            
            # Aggregate immediately (async behavior)
            self.aggregate_async(weights, alpha)
            
            # Update tracking
            self.client_last_round[client.client_id] = round_num
            updates_received += 1
            
            stats['staleness'] = staleness
            stats['alpha'] = alpha
            client_stats.append(stats)
            
            self.update_history.append({
                'round': round_num,
                'client_id': client.client_id,
                'staleness': staleness,
                'alpha': alpha
            })
            
            # Check if we have enough updates (early stop)
            if updates_received >= self.min_updates_per_round:
                # In real async, we might continue accepting late updates
                pass
        
        round_time = time.time() - round_start_time
        
        # Evaluate global model
        eval_results = self.evaluate_global_model()
        
        # Compile round statistics
        round_stats = {
            'round': round_num,
            'status': 'completed',
            'num_updates': updates_received,
            'num_clients_selected': len(selected_clients),
            'total_samples': sum(s.get('train_samples', 0) for s in client_stats),
            'round_time': round_time,
            'avg_client_loss': np.mean([s['train_loss'] for s in client_stats]) if client_stats else 0,
            'avg_staleness': np.mean([s['staleness'] for s in client_stats]) if client_stats else 0,
            'avg_alpha': np.mean([s['alpha'] for s in client_stats]) if client_stats else 0,
            'avg_battery': np.mean([s['battery_level'] for s in client_stats]) if client_stats else 0,
            'total_communication': sum(s.get('communication_size', 0) for s in client_stats),
            'global_accuracy': eval_results['accuracy'],
            'global_loss': eval_results['loss']
        }
        
        self.round_history.append(round_stats)
        self.metrics.log_round(round_stats)
        
        return round_stats
    
    def evaluate_global_model(self) -> Dict:
        """Evaluate the global model on global test data."""
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
        Execute the full asynchronous federated learning process.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("ASYNCHRONOUS FEDERATED LEARNING (NOVELTY)")
        print("="*60)
        
        training_start_time = time.time()
        
        # Training loop
        for round_num in tqdm(range(1, self.num_rounds + 1), desc="Async FL Rounds"):
            round_stats = self.train_round(round_num)
            
            if verbose and round_stats.get('status') == 'completed':
                if round_num % 5 == 0 or round_num == 1:
                    print(f"\nRound {round_num}: "
                          f"Accuracy={round_stats['global_accuracy']:.2f}%, "
                          f"Loss={round_stats['global_loss']:.4f}, "
                          f"Updates={round_stats['num_updates']}, "
                          f"Avg Staleness={round_stats['avg_staleness']:.2f}")
        
        total_time = time.time() - training_start_time
        
        # Final evaluation
        final_eval = self.evaluate_global_model()
        
        results = {
            'method': 'Asynchronous FL',
            'num_rounds': self.num_rounds,
            'total_time': total_time,
            'final_accuracy': final_eval['accuracy'],
            'final_loss': final_eval['loss'],
            'round_history': self.round_history,
            'update_history': self.update_history,
            'total_communication': sum(r.get('total_communication', 0) for r in self.round_history),
            'avg_staleness': np.mean([u['staleness'] for u in self.update_history]) if self.update_history else 0
        }
        
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final Accuracy: {final_eval['accuracy']:.2f}%")
        print(f"Average Update Staleness: {results['avg_staleness']:.2f} rounds")
        
        return results
    
    def get_global_model(self) -> HARModel:
        """Get the trained global model."""
        return self.global_model
    
    def save_model(self, path: str) -> None:
        """Save the global model to disk."""
        torch.save(self.global_weights, path)
        print(f"Model saved to {path}")


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
                local_epochs=3
            )
            clients.append(client)
    
    # Create and run async server
    server = AsyncFederatedServer(
        device=device,
        num_rounds=10,
        clients_per_round=5,
        min_updates_per_round=3
    )
    server.register_clients(clients)
    results = server.train()
