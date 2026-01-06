"""
Evaluation Metrics Module
=========================
Comprehensive metrics tracking and visualization for
federated learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class MetricsTracker:
    """
    Tracks and logs metrics during federated learning.
    
    Metrics tracked:
    - Accuracy (per round and final)
    - Loss (training and test)
    - Training time
    - Communication cost
    - Energy consumption
    - Client participation
    """
    
    def __init__(self, experiment_name: str = "experiment"):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name for this experiment
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Round-level metrics
        self.rounds: List[Dict] = []
        
        # Aggregated metrics
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.time_history: List[float] = []
        self.communication_history: List[float] = []
        self.energy_history: List[float] = []
        
    def log_round(self, round_stats: Dict) -> None:
        """
        Log metrics for one round.
        
        Args:
            round_stats: Dictionary with round metrics
        """
        self.rounds.append(round_stats)
        
        # Extract key metrics
        if 'global_accuracy' in round_stats:
            self.accuracy_history.append(round_stats['global_accuracy'])
        if 'global_loss' in round_stats:
            self.loss_history.append(round_stats['global_loss'])
        if 'round_time' in round_stats:
            self.time_history.append(round_stats['round_time'])
        if 'total_communication' in round_stats:
            self.communication_history.append(round_stats['total_communication'])
    
    def get_summary(self) -> Dict:
        """
        Get summary of tracked metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {
            'experiment_name': self.experiment_name,
            'num_rounds': len(self.rounds),
            'total_time': sum(self.time_history) if self.time_history else 0,
        }
        
        if self.accuracy_history:
            summary['final_accuracy'] = self.accuracy_history[-1]
            summary['best_accuracy'] = max(self.accuracy_history)
            summary['avg_accuracy'] = np.mean(self.accuracy_history)
        
        if self.loss_history:
            summary['final_loss'] = self.loss_history[-1]
            summary['best_loss'] = min(self.loss_history)
        
        if self.communication_history:
            summary['total_communication_bytes'] = sum(self.communication_history)
            summary['total_communication_mb'] = sum(self.communication_history) / (1024 * 1024)
        
        return summary
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        data = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'summary': self.get_summary(),
            'rounds': self.rounds,
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'communication_history': self.communication_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MetricsTracker':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(data.get('experiment_name', 'loaded'))
        tracker.rounds = data.get('rounds', [])
        tracker.accuracy_history = data.get('accuracy_history', [])
        tracker.loss_history = data.get('loss_history', [])
        tracker.time_history = data.get('time_history', [])
        tracker.communication_history = data.get('communication_history', [])
        
        return tracker


class ExperimentComparator:
    """
    Compares results across multiple experiments.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
    
    def add_experiment(self, name: str, results: Dict) -> None:
        """Add experiment results."""
        self.experiments[name] = results
    
    def compare(self) -> Dict:
        """
        Compare all experiments.
        
        Returns:
            Comparison dictionary
        """
        comparison = {
            'experiments': list(self.experiments.keys()),
            'metrics': {}
        }
        
        # Compare final accuracy
        accuracies = {
            name: res.get('final_accuracy', 0) 
            for name, res in self.experiments.items()
        }
        comparison['metrics']['accuracy'] = accuracies
        comparison['best_accuracy'] = max(accuracies, key=accuracies.get)
        
        # Compare training time
        times = {
            name: res.get('total_time', 0) 
            for name, res in self.experiments.items()
        }
        comparison['metrics']['training_time'] = times
        comparison['fastest'] = min(times, key=times.get)
        
        # Compare communication cost
        comm = {
            name: res.get('total_communication', 0) 
            for name, res in self.experiments.items()
        }
        comparison['metrics']['communication'] = comm
        
        return comparison
    
    def print_comparison(self) -> None:
        """Print formatted comparison table."""
        comparison = self.compare()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPARISON")
        print("="*70)
        
        # Header
        print(f"{'Experiment':<25} {'Accuracy':<15} {'Time (s)':<15} {'Comm (MB)':<15}")
        print("-"*70)
        
        # Data rows
        for name in comparison['experiments']:
            acc = comparison['metrics']['accuracy'].get(name, 0)
            time_val = comparison['metrics']['training_time'].get(name, 0)
            comm = comparison['metrics']['communication'].get(name, 0) / (1024 * 1024)
            
            print(f"{name:<25} {acc:<15.2f} {time_val:<15.2f} {comm:<15.2f}")
        
        print("-"*70)
        print(f"Best Accuracy: {comparison['best_accuracy']}")
        print(f"Fastest: {comparison['fastest']}")
        print("="*70)


def plot_training_curves(
    experiments: Dict[str, Dict],
    save_path: str = None
) -> None:
    """
    Plot training curves for multiple experiments.
    
    Args:
        experiments: Dictionary mapping experiment name to results
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Plot 1: Accuracy over rounds
    ax1 = axes[0, 0]
    for (name, results), color in zip(experiments.items(), colors):
        if 'round_history' in results:
            accuracies = [r.get('global_accuracy', r.get('test_accuracy', 0)) 
                         for r in results['round_history']]
            ax1.plot(accuracies, label=name, color=color, linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy over Training Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss over rounds
    ax2 = axes[0, 1]
    for (name, results), color in zip(experiments.items(), colors):
        if 'round_history' in results:
            losses = [r.get('global_loss', r.get('test_loss', 0)) 
                     for r in results['round_history']]
            ax2.plot(losses, label=name, color=color, linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Training Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Communication cost
    ax3 = axes[1, 0]
    for (name, results), color in zip(experiments.items(), colors):
        if 'round_history' in results:
            comm = [r.get('total_communication', 0) / 1024 
                   for r in results['round_history']]
            cumulative_comm = np.cumsum(comm)
            ax3.plot(cumulative_comm, label=name, color=color, linewidth=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative Communication (KB)')
    ax3.set_title('Communication Cost over Rounds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics comparison
    ax4 = axes[1, 1]
    names = list(experiments.keys())
    final_accuracies = [experiments[n].get('final_accuracy', 0) for n in names]
    x = np.arange(len(names))
    bars = ax4.bar(x, final_accuracies, color=colors[:len(names)])
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel('Final Accuracy (%)')
    ax4.set_title('Final Accuracy Comparison')
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_energy_analysis(
    client_stats: List[Dict],
    save_path: str = None
) -> None:
    """
    Plot energy consumption analysis.
    
    Args:
        client_stats: List of client statistics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    client_ids = [s['client_id'] for s in client_stats]
    energy_consumed = [s.get('total_consumed', 0) for s in client_stats]
    battery_levels = [s.get('current_battery', 0) for s in client_stats]
    
    # Plot 1: Energy consumption per client
    ax1 = axes[0]
    ax1.bar(range(len(client_ids)), energy_consumed, color='coral')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Energy Consumed (%)')
    ax1.set_title('Energy Consumption per Client')
    ax1.set_xticks(range(len(client_ids)))
    ax1.set_xticklabels(client_ids, rotation=45)
    
    # Plot 2: Remaining battery levels
    ax2 = axes[1]
    ax2.bar(range(len(client_ids)), battery_levels, color='lightgreen')
    ax2.axhline(y=20, color='red', linestyle='--', label='Min Battery Threshold')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Battery Level (%)')
    ax2.set_title('Remaining Battery per Client')
    ax2.set_xticks(range(len(client_ids)))
    ax2.set_xticklabels(client_ids, rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_fl_efficiency(
    fl_results: Dict,
    centralized_results: Dict
) -> Dict:
    """
    Calculate efficiency metrics comparing FL to centralized.
    
    Args:
        fl_results: Federated learning results
        centralized_results: Centralized learning results
        
    Returns:
        Efficiency metrics
    """
    fl_acc = fl_results.get('final_accuracy', 0)
    central_acc = centralized_results.get('final_accuracy', 0)
    
    fl_time = fl_results.get('total_time', 1)
    central_time = centralized_results.get('total_time', 1)
    
    return {
        'accuracy_gap': central_acc - fl_acc,
        'accuracy_ratio': fl_acc / max(central_acc, 1) * 100,
        'time_ratio': fl_time / max(central_time, 1),
        'privacy_preserved': True,  # FL preserves data privacy
        'communication_overhead': fl_results.get('total_communication', 0)
    }


if __name__ == "__main__":
    # Test metrics module
    print("Metrics Module Test")
    
    # Create sample data
    tracker = MetricsTracker("test_experiment")
    
    for i in range(10):
        tracker.log_round({
            'round': i + 1,
            'global_accuracy': 50 + i * 4 + np.random.randn() * 2,
            'global_loss': 2.0 - i * 0.15 + np.random.randn() * 0.1,
            'round_time': 5 + np.random.randn(),
            'total_communication': 100000 + np.random.randint(-10000, 10000)
        })
    
    summary = tracker.get_summary()
    print("\nMetrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
