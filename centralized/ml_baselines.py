"""
Traditional ML Baselines for HAR Dataset
=========================================
Compares traditional ML models with deep learning approaches.
Shows why Federated Learning with neural networks is valuable.

Usage:
    python centralized/ml_baselines.py
"""

import numpy as np
import time
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from data.har_loader import HARDataLoader


class MLBaselines:
    """
    Traditional Machine Learning baselines for HAR classification.
    
    Purpose: Compare with deep learning to show:
    1. DL achieves similar/better accuracy
    2. But ONLY DL can be used in Federated Learning
    3. FL preserves privacy while maintaining accuracy
    """
    
    def __init__(self):
        """Initialize ML baselines."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.results: Dict[str, Dict] = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load HAR dataset for ML training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        har_loader = HARDataLoader()
        
        # Load raw data
        train_features, train_labels, _ = har_loader.load_data("train")
        test_features, test_labels, _ = har_loader.load_data("test")
        
        # Standardize features (important for SVM, LogReg, KNN)
        X_train = self.scaler.fit_transform(train_features)
        X_test = self.scaler.transform(test_features)
        
        return X_train, X_test, train_labels, test_labels
    
    def train_and_evaluate(
        self, 
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Train and evaluate a single model.
        
        Args:
            model_name: Name of the model to train
            X_train, X_test: Feature matrices
            y_train, y_test: Label arrays
            verbose: Print progress
            
        Returns:
            Dictionary with results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        if verbose:
            print(f"\nTraining {model_name}...", end=" ")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        if verbose:
            print(f"Done! Accuracy: {accuracy:.2f}% (Time: {train_time:.2f}s)")
        
        result = {
            'model': model_name,
            'accuracy': accuracy,
            'training_time': train_time,
            'predictions': y_pred
        }
        
        self.results[model_name] = result
        return result
    
    def run_all_baselines(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Run all ML baseline experiments.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary of all results
        """
        print("\n" + "="*70)
        print("TRADITIONAL ML BASELINES FOR HAR DATASET")
        print("="*70)
        
        # Load data
        print("\nLoading and preprocessing data...")
        X_train, X_test, y_train, y_test = self.load_data()
        print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
        print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
        
        # Train all models
        print("\n" + "-"*70)
        print("TRAINING MODELS")
        print("-"*70)
        
        for model_name in self.models.keys():
            self.train_and_evaluate(
                model_name, X_train, X_test, y_train, y_test, verbose
            )
        
        # Print comparison table
        self.print_comparison_table()
        
        # Print detailed report for best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        self.print_detailed_report(best_model, y_test)
        
        return self.results
    
    def print_comparison_table(self) -> None:
        """Print formatted comparison table."""
        print("\n" + "="*70)
        print("ML BASELINES COMPARISON")
        print("="*70)
        
        print(f"{'Model':<25} {'Accuracy (%)':<15} {'Time (s)':<15}")
        print("-"*70)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        for model_name, result in sorted_results:
            print(f"{model_name:<25} {result['accuracy']:<15.2f} {result['training_time']:<15.2f}")
        
        print("-"*70)
        
        # Best model
        best = sorted_results[0]
        print(f"\nBest ML Model: {best[0]} ({best[1]['accuracy']:.2f}%)")
        print("="*70)
    
    def print_detailed_report(self, model_name: str, y_test: np.ndarray) -> None:
        """Print detailed classification report for a model."""
        if model_name not in self.results:
            return
        
        result = self.results[model_name]
        y_pred = result['predictions']
        
        print(f"\n" + "-"*70)
        print(f"DETAILED REPORT: {model_name}")
        print("-"*70)
        
        # Activity labels
        target_names = [
            'WALKING', 'WALKING_UP', 'WALKING_DOWN', 
            'SITTING', 'STANDING', 'LAYING'
        ]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
    
def run_ml_baselines() -> Dict:
    """
    Run all ML baseline experiments.
    
    Returns:
        Dictionary with all results
    """
    baselines = MLBaselines()
    results = baselines.run_all_baselines()
    return results


if __name__ == "__main__":
    results = run_ml_baselines()
    
    # Print summary
    print("\n" + "="*70)
    print("ML BASELINES COMPLETED")
    print("="*70)
    print("\nTo see full comparison with DL and FL, run:")
    print("  python run_experiments.py")
    print("="*70)
