"""
Adaptive Client Selection (NOVELTY MODULE)
==========================================
Selects clients dynamically based on battery level and network latency.
This improves training efficiency and handles heterogeneous IoT devices.
"""

import numpy as np
from typing import List, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdaptiveClientSelector:
    """
    Adaptive client selection based on device characteristics.
    
    NOVELTY: Considers battery level and network latency to select
    optimal clients for each round, improving efficiency and convergence.
    
    Selection Score = battery_weight * battery_score + latency_weight * latency_score + data_weight * data_score
    """
    
    def __init__(
        self,
        min_battery: float = 20.0,
        max_latency: float = 2.0,
        battery_weight: float = 0.4,
        latency_weight: float = 0.3,
        data_weight: float = 0.3,
        exploration_rate: float = 0.1
    ):
        """
        Initialize the adaptive client selector.
        
        Args:
            min_battery: Minimum battery level to consider a client
            max_latency: Maximum acceptable latency in seconds
            battery_weight: Weight for battery score in selection
            latency_weight: Weight for latency score in selection
            data_weight: Weight for data size score in selection
            exploration_rate: Probability of random selection (exploration)
        """
        self.min_battery = min_battery
        self.max_latency = max_latency
        self.battery_weight = battery_weight
        self.latency_weight = latency_weight
        self.data_weight = data_weight
        self.exploration_rate = exploration_rate
        
        # Track client performance history
        self.client_history: Dict[int, Dict] = {}
        self.selection_history: List[Dict] = []
        
    def calculate_client_score(self, client) -> float:
        """
        Calculate selection score for a client.
        
        Higher score = better candidate for selection
        
        Args:
            client: IoTClient instance
            
        Returns:
            Selection score between 0 and 1
        """
        # Battery score (higher is better)
        battery_level = client.get_battery_level()
        if battery_level < self.min_battery:
            return 0.0  # Exclude low battery clients
        battery_score = battery_level / 100.0
        
        # Latency score (lower latency is better)
        latency = client.get_latency()
        if latency > self.max_latency:
            latency_score = 0.0
        else:
            latency_score = 1.0 - (latency / self.max_latency)
        
        # Data size score (more data is better for learning)
        max_data_size = 500  # Approximate max samples per client
        data_score = min(client.train_size / max_data_size, 1.0)
        
        # Combine scores
        total_score = (
            self.battery_weight * battery_score +
            self.latency_weight * latency_score +
            self.data_weight * data_score
        )
        
        return total_score
    
    def select_clients(
        self,
        clients: List,
        num_to_select: int,
        round_num: int
    ) -> List:
        """
        Select clients for the current round.
        
        Uses a combination of:
        - Exploitation: Select clients with highest scores
        - Exploration: Randomly select some clients to gather information
        
        Args:
            clients: List of all available clients
            num_to_select: Number of clients to select
            round_num: Current round number
            
        Returns:
            List of selected clients
        """
        if len(clients) == 0:
            return []
        
        # Calculate scores for all clients
        client_scores = []
        for client in clients:
            score = self.calculate_client_score(client)
            client_scores.append({
                'client': client,
                'score': score,
                'battery': client.get_battery_level(),
                'latency': client.get_latency()
            })
        
        # Filter out clients with zero score
        eligible_clients = [c for c in client_scores if c['score'] > 0]
        
        if len(eligible_clients) == 0:
            # If no eligible clients, select randomly from all
            num_select = min(num_to_select, len(clients))
            return list(np.random.choice(clients, size=num_select, replace=False))
        
        num_to_select = min(num_to_select, len(eligible_clients))
        
        # Determine exploration vs exploitation
        num_explore = max(1, int(num_to_select * self.exploration_rate))
        num_exploit = num_to_select - num_explore
        
        # Sort by score for exploitation
        sorted_clients = sorted(eligible_clients, key=lambda x: x['score'], reverse=True)
        
        # Exploit: Select top-scoring clients
        exploit_selection = [c['client'] for c in sorted_clients[:num_exploit]]
        
        # Explore: Randomly select from remaining clients
        remaining = [c['client'] for c in sorted_clients[num_exploit:]]
        if len(remaining) > 0 and num_explore > 0:
            explore_selection = list(np.random.choice(
                remaining,
                size=min(num_explore, len(remaining)),
                replace=False
            ))
        else:
            explore_selection = []
        
        selected = exploit_selection + explore_selection
        
        # Log selection
        self.selection_history.append({
            'round': round_num,
            'num_selected': len(selected),
            'num_eligible': len(eligible_clients),
            'avg_score': np.mean([c['score'] for c in client_scores if c['score'] > 0]),
            'selected_ids': [c.client_id for c in selected]
        })
        
        return selected
    
    def update_client_history(
        self,
        client_id: int,
        round_num: int,
        performance: Dict
    ) -> None:
        """
        Update performance history for a client.
        
        Args:
            client_id: Client identifier
            round_num: Round number
            performance: Dictionary with performance metrics
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = {
                'rounds_participated': [],
                'performance': []
            }
        
        self.client_history[client_id]['rounds_participated'].append(round_num)
        self.client_history[client_id]['performance'].append(performance)
    
    def get_selection_statistics(self) -> Dict:
        """Get statistics about client selection."""
        if not self.selection_history:
            return {}
        
        return {
            'total_rounds': len(self.selection_history),
            'avg_clients_selected': np.mean([s['num_selected'] for s in self.selection_history]),
            'avg_score': np.mean([s['avg_score'] for s in self.selection_history]),
            'unique_clients_used': len(set(
                cid for s in self.selection_history for cid in s['selected_ids']
            ))
        }


class PriorityClientSelector:
    """
    Alternative selection strategy using priority queues.
    Clients that haven't participated recently get higher priority.
    """
    
    def __init__(
        self,
        min_battery: float = 20.0,
        fairness_weight: float = 0.3
    ):
        self.min_battery = min_battery
        self.fairness_weight = fairness_weight
        self.last_participation: Dict[int, int] = {}
    
    def select_clients(
        self,
        clients: List,
        num_to_select: int,
        round_num: int
    ) -> List:
        """Select clients with fairness consideration."""
        scored_clients = []
        
        for client in clients:
            if client.get_battery_level() < self.min_battery:
                continue
            
            # Rounds since last participation
            last_round = self.last_participation.get(client.client_id, 0)
            fairness_score = min((round_num - last_round) / 10.0, 1.0)
            
            # Combine with battery score
            battery_score = client.get_battery_level() / 100.0
            
            total_score = (
                (1 - self.fairness_weight) * battery_score +
                self.fairness_weight * fairness_score
            )
            
            scored_clients.append((client, total_score))
        
        # Sort and select
        scored_clients.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in scored_clients[:num_to_select]]
        
        # Update participation record
        for client in selected:
            self.last_participation[client.client_id] = round_num
        
        return selected


if __name__ == "__main__":
    # Test client selection
    print("Adaptive Client Selection Module")
    print("This module is tested through the federated learning experiments")
