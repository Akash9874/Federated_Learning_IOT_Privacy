"""
HAR Dataset Loader for Federated Learning
==========================================
Downloads and processes the UCI Human Activity Recognition Dataset.
Each user (subject) is treated as a separate IoT device for non-IID data distribution.

Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class HARDataset(Dataset):
    """PyTorch Dataset for HAR data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class HARDataLoader:
    """
    Loads and preprocesses the UCI HAR dataset.
    Maps each subject (user) to a simulated IoT device for federated learning.
    """
    
    DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    
    # Activity labels mapping
    ACTIVITIES = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    def __init__(self, data_dir: str = "./data/har_dataset"):
        """
        Initialize the HAR data loader.
        
        Args:
            data_dir: Directory to store/load the dataset
        """
        self.data_dir = data_dir
        # Check if dataset is directly in data_dir or in UCI HAR Dataset subfolder
        if os.path.exists(os.path.join(data_dir, "train")):
            self.dataset_path = data_dir
        else:
            self.dataset_path = os.path.join(data_dir, "UCI HAR Dataset")
        self.num_features = 561  # Number of features in HAR dataset
        self.num_classes = 6     # Number of activity classes
        
    def download_dataset(self) -> None:
        """Download the UCI HAR dataset if not already present."""
        if os.path.exists(self.dataset_path):
            print("Dataset already exists. Skipping download.")
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "har_dataset.zip")
        
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    
    def load_data(self, subset: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load features, labels, and subject IDs for a given subset.
        
        Args:
            subset: Either 'train' or 'test'
            
        Returns:
            Tuple of (features, labels, subject_ids)
        """
        subset_path = os.path.join(self.dataset_path, subset)
        
        # Load features (X)
        features_path = os.path.join(subset_path, f"X_{subset}.txt")
        features = pd.read_csv(features_path, sep=r'\s+', header=None).values
        
        # Load labels (y) - Convert to 0-indexed
        labels_path = os.path.join(subset_path, f"y_{subset}.txt")
        labels = pd.read_csv(labels_path, header=None).values.flatten() - 1
        
        # Load subject IDs
        subjects_path = os.path.join(subset_path, f"subject_{subset}.txt")
        subject_ids = pd.read_csv(subjects_path, header=None).values.flatten()
        
        return features, labels, subject_ids
    
    def get_client_data(self) -> Dict[int, Dict[str, HARDataset]]:
        """
        Split data by subject ID to simulate IoT devices.
        Each subject becomes one federated learning client.
        
        This creates a NON-IID data distribution as each user has different
        activity patterns and proportions.
        
        Returns:
            Dictionary mapping client_id to {'train': Dataset, 'test': Dataset}
        """
        self.download_dataset()
        
        # Load train and test data
        train_features, train_labels, train_subjects = self.load_data("train")
        test_features, test_labels, test_subjects = self.load_data("test")
        
        # Get unique subject IDs (there are 30 subjects in HAR dataset)
        all_subjects = np.unique(np.concatenate([train_subjects, test_subjects]))
        
        client_data = {}
        
        for subject_id in all_subjects:
            # Get indices for this subject
            train_mask = train_subjects == subject_id
            test_mask = test_subjects == subject_id
            
            # Create datasets for this client
            if np.sum(train_mask) > 0:
                train_dataset = HARDataset(
                    train_features[train_mask],
                    train_labels[train_mask]
                )
            else:
                train_dataset = None
                
            if np.sum(test_mask) > 0:
                test_dataset = HARDataset(
                    test_features[test_mask],
                    test_labels[test_mask]
                )
            else:
                test_dataset = None
            
            client_data[subject_id] = {
                'train': train_dataset,
                'test': test_dataset,
                'train_size': np.sum(train_mask),
                'test_size': np.sum(test_mask)
            }
            
        print(f"Data distributed to {len(client_data)} clients (IoT devices)")
        self._print_data_distribution(client_data, train_labels, train_subjects)
        
        return client_data
    
    def _print_data_distribution(
        self, 
        client_data: Dict, 
        labels: np.ndarray, 
        subjects: np.ndarray
    ) -> None:
        """Print the non-IID data distribution across clients."""
        print("\n" + "="*60)
        print("NON-IID DATA DISTRIBUTION ACROSS IoT DEVICES")
        print("="*60)
        
        for client_id, data in client_data.items():
            if data['train'] is not None:
                client_labels = labels[subjects == client_id]
                unique, counts = np.unique(client_labels, return_counts=True)
                dist = {self.ACTIVITIES[u+1]: c for u, c in zip(unique, counts)}
                print(f"Client {client_id}: {data['train_size']} samples - {dist}")
        
        print("="*60 + "\n")
    
    def get_centralized_data(self) -> Tuple[HARDataset, HARDataset]:
        """
        Get full dataset for centralized training baseline.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        self.download_dataset()
        
        train_features, train_labels, _ = self.load_data("train")
        test_features, test_labels, _ = self.load_data("test")
        
        train_dataset = HARDataset(train_features, train_labels)
        test_dataset = HARDataset(test_features, test_labels)
        
        print(f"Centralized data: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        return train_dataset, test_dataset
    
    def get_data_loaders(
        self, 
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get DataLoaders for centralized training.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_dataset, test_dataset = self.get_centralized_data()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader


def get_feature_names() -> List[str]:
    """Return the list of feature names from the HAR dataset."""
    # The dataset has 561 features derived from accelerometer and gyroscope
    # This is a simplified list - full list is in features.txt of the dataset
    return [f"feature_{i}" for i in range(561)]


if __name__ == "__main__":
    # Test the data loader
    loader = HARDataLoader()
    
    # Test centralized data loading
    train_data, test_data = loader.get_centralized_data()
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Test federated data loading
    client_data = loader.get_client_data()
    print(f"Number of clients: {len(client_data)}")
