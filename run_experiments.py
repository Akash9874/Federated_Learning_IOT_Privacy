"""
Main Experiment Runner
======================
Executes all experiments: Centralized, Sync FL, and Async FL.
Compares results and generates reports.

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --run-async        # Run only async FL
    python run_experiments.py --enable-dp        # Run with differential privacy
    python run_experiments.py --num-rounds 50    # Run for 50 rounds
"""

import torch
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.har_loader import HARDataLoader
from centralized.centralized_train import CentralizedTrainer
from federated.model import create_model
from federated.client import IoTClient
from federated.server_sync import SyncFederatedServer
from federated.server_async import AsyncFederatedServer
from evaluation.metrics import (
    MetricsTracker, 
    ExperimentComparator, 
    plot_training_curves,
    calculate_fl_efficiency
)


def setup_clients(
    client_data: dict,
    device: torch.device,
    local_epochs: int = 3,
    enable_dp: bool = False,
    enable_compression: bool = False
) -> list:
    """
    Create IoT clients from client data.
    
    Args:
        client_data: Dictionary of client datasets
        device: PyTorch device
        local_epochs: Number of local training epochs
        enable_dp: Enable differential privacy
        enable_compression: Enable model compression
        
    Returns:
        List of IoTClient instances
    """
    clients = []
    
    for client_id, data in client_data.items():
        if data['train'] is None:
            continue
        
        # Simulate heterogeneous devices with varying battery and latency
        initial_battery = np.random.uniform(60, 100)
        latency_range = (np.random.uniform(0.1, 0.5), np.random.uniform(0.5, 2.0))
        
        client = IoTClient(
            client_id=client_id,
            train_dataset=data['train'],
            test_dataset=data['test'],
            device=device,
            local_epochs=local_epochs,
            batch_size=32,
            learning_rate=0.01,
            initial_battery=initial_battery,
            latency_range=latency_range,
            enable_dp=enable_dp,
            dp_epsilon=1.0,
            enable_compression=enable_compression,
            compression_ratio=0.5
        )
        clients.append(client)
    
    return clients


def run_centralized_experiment(
    device: torch.device,
    num_epochs: int = 30
) -> dict:
    """Run centralized learning baseline."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: CENTRALIZED LEARNING BASELINE")
    print("="*70)
    
    trainer = CentralizedTrainer(
        device=device,
        num_epochs=num_epochs,
        batch_size=64,
        learning_rate=0.001
    )
    
    results = trainer.train(verbose=True)
    return results


def run_sync_fl_experiment(
    device: torch.device,
    clients: list,
    num_rounds: int = 30,
    clients_per_round: int = 10,
    test_dataset = None
) -> dict:
    """Run synchronous federated learning."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SYNCHRONOUS FEDERATED LEARNING (FedAvg)")
    print("="*70)
    
    # Reset client batteries for fair comparison
    for client in clients:
        client.reset_battery()
    
    server = SyncFederatedServer(
        device=device,
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        min_clients=3,
        adaptive_selection=True
    )
    
    server.register_clients(clients)
    
    # Set global test dataset for evaluation
    if test_dataset is not None:
        server.set_test_dataset(test_dataset)
    
    results = server.train(verbose=True)
    
    return results


def run_async_fl_experiment(
    device: torch.device,
    clients: list,
    num_rounds: int = 30,
    clients_per_round: int = 10,
    test_dataset = None
) -> dict:
    """Run asynchronous federated learning (NOVELTY)."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: ASYNCHRONOUS FEDERATED LEARNING (NOVELTY)")
    print("="*70)
    
    # Reset client batteries for fair comparison
    for client in clients:
        client.reset_battery()
    
    server = AsyncFederatedServer(
        device=device,
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        min_updates_per_round=3,
        base_alpha=0.5,
        staleness_discount=0.9,
        adaptive_selection=True
    )
    
    server.register_clients(clients)
    
    # Set global test dataset for evaluation
    if test_dataset is not None:
        server.set_test_dataset(test_dataset)
    
    results = server.train(verbose=True)
    
    return results


def run_all_experiments(args):
    """Run all experiments and compare results."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading HAR dataset...")
    har_loader = HARDataLoader(data_dir=args.data_dir)
    client_data = har_loader.get_client_data()
    
    # Get global test dataset for FL evaluation
    _, test_dataset = har_loader.get_centralized_data()
    
    # Create clients
    print("\nSetting up IoT clients...")
    clients = setup_clients(
        client_data=client_data,
        device=device,
        local_epochs=args.local_epochs,
        enable_dp=args.enable_dp,
        enable_compression=args.enable_compression
    )
    print(f"Created {len(clients)} clients")
    
    # Results storage
    all_results = {}
    
    # Experiment 1: Centralized Learning
    if args.run_centralized:
        central_results = run_centralized_experiment(
            device=device,
            num_epochs=args.num_rounds
        )
        all_results['Centralized'] = central_results
    
    # Experiment 2: Synchronous FL
    if args.run_sync:
        sync_results = run_sync_fl_experiment(
            device=device,
            clients=clients,
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            test_dataset=test_dataset
        )
        all_results['Sync FL'] = sync_results
    
    # Experiment 3: Asynchronous FL
    if args.run_async:
        async_results = run_async_fl_experiment(
            device=device,
            clients=clients,
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            test_dataset=test_dataset
        )
        all_results['Async FL'] = async_results
    
    # Compare results
    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    
    comparator = ExperimentComparator()
    for name, results in all_results.items():
        comparator.add_experiment(name, results)
    
    comparator.print_comparison()
    
    # Calculate efficiency metrics
    if 'Centralized' in all_results and 'Sync FL' in all_results:
        efficiency = calculate_fl_efficiency(
            all_results['Sync FL'],
            all_results['Centralized']
        )
        print("\nFederated Learning Efficiency:")
        print(f"  Accuracy Gap: {efficiency['accuracy_gap']:.2f}%")
        print(f"  Accuracy Ratio: {efficiency['accuracy_ratio']:.2f}%")
        print(f"  Privacy Preserved: {efficiency['privacy_preserved']}")
    
    # Plot results
    if args.plot and len(all_results) > 0:
        try:
            plot_training_curves(
                all_results,
                save_path=os.path.join(output_dir, 'training_curves.png')
            )
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    # Save results
    results_file = os.path.join(output_dir, 'experiment_results.json')
    save_results = {}
    for name, results in all_results.items():
        save_results[name] = {
            'method': results.get('method', name),
            'final_accuracy': results.get('final_accuracy', 0),
            'final_loss': results.get('final_loss', 0),
            'total_time': results.get('total_time', 0),
            'total_communication': results.get('total_communication', 0)
        }
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Final Accuracy: {results.get('final_accuracy', 0):.2f}%")
        print(f"  Training Time: {results.get('total_time', 0):.2f}s")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated Learning for IoT - Experiment Runner'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='./data/har_dataset',
        help='Directory for HAR dataset'
    )
    
    # Training arguments
    parser.add_argument(
        '--num-rounds', 
        type=int, 
        default=30,
        help='Number of training rounds/epochs'
    )
    parser.add_argument(
        '--local-epochs', 
        type=int, 
        default=3,
        help='Number of local epochs per round'
    )
    parser.add_argument(
        '--clients-per-round', 
        type=int, 
        default=10,
        help='Number of clients per FL round'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    
    # Experiment selection
    parser.add_argument(
        '--run-centralized', 
        action='store_true', 
        default=True,
        help='Run centralized learning experiment'
    )
    parser.add_argument(
        '--run-sync', 
        action='store_true', 
        default=True,
        help='Run synchronous FL experiment'
    )
    parser.add_argument(
        '--run-async', 
        action='store_true', 
        default=True,
        help='Run asynchronous FL experiment'
    )
    parser.add_argument(
        '--skip-centralized',
        action='store_true',
        default=False,
        help='Skip centralized learning experiment'
    )
    parser.add_argument(
        '--skip-sync',
        action='store_true',
        default=False,
        help='Skip synchronous FL experiment'
    )
    parser.add_argument(
        '--skip-async',
        action='store_true',
        default=False,
        help='Skip asynchronous FL experiment'
    )
    
    # Novelty features
    parser.add_argument(
        '--enable-dp', 
        action='store_true', 
        default=False,
        help='Enable differential privacy'
    )
    parser.add_argument(
        '--enable-compression', 
        action='store_true', 
        default=False,
        help='Enable model compression'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--plot', 
        action='store_true', 
        default=True,
        help='Generate plots'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        default=False,
        help='Disable plot generation'
    )
    
    args = parser.parse_args()
    
    # Handle skip flags
    if args.skip_centralized:
        args.run_centralized = False
    if args.skip_sync:
        args.run_sync = False
    if args.skip_async:
        args.run_async = False
    if args.no_plot:
        args.plot = False
    
    # Run experiments
    results = run_all_experiments(args)
    
    return results


if __name__ == "__main__":
    main()
