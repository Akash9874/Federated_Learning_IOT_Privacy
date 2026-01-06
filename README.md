# Asynchronous, Energy-Aware, Privacy-Preserving Federated Learning Framework for IoT Devices using AWS IoT Core

## 📋 Project Overview

This is a comprehensive **Final Year Major Project** implementing an end-to-end Federated Learning (FL) framework designed for IoT devices. The project demonstrates advanced concepts in distributed machine learning, privacy preservation, and resource-efficient training.

### Key Features

- **Synchronous Federated Learning (FedAvg)**: Classic FL where server waits for all clients
- **Asynchronous Federated Learning (NOVELTY)**: Server aggregates updates immediately as they arrive
- **Adaptive Client Selection**: Battery and latency-aware client selection
- **Energy-Aware Training**: Dynamic epoch adjustment based on device battery
- **Model Compression**: Efficient communication with compressed model updates
- **Differential Privacy**: Device-level privacy protection with Gaussian noise
- **AWS IoT Core Integration**: MQTT-based secure communication

### Dataset

**UCI Human Activity Recognition (HAR) Dataset**
- 30 subjects performing 6 activities
- Each subject treated as one IoT device (non-IID distribution)
- 561 features from smartphone sensors

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FEDERATED LEARNING SERVER                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   FedAvg        │  │   Async FL      │  │   Model         │  │
│  │   Aggregation   │  │   Aggregation   │  │   Broadcasting  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            │        AWS IoT Core (MQTT)                │
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼──────────┐
│           ▼                     ▼                     ▼          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  IoT Client │  │  IoT Client │  │  IoT Client │   ...        │
│  │  (Device 1) │  │  (Device 2) │  │  (Device N) │              │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤              │
│  │ Local Data  │  │ Local Data  │  │ Local Data  │              │
│  │ Local Train │  │ Local Train │  │ Local Train │              │
│  │ DP Noise    │  │ DP Noise    │  │ DP Noise    │              │
│  │ Compression │  │ Compression │  │ Compression │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Federated_IoT_Major_Project/
│
├── data/
│   └── har_loader.py          # HAR dataset loader with non-IID split
│
├── centralized/
│   └── centralized_train.py   # Centralized learning baseline
│
├── federated/
│   ├── model.py               # PyTorch neural network model
│   ├── client.py              # IoT client implementation
│   ├── server_sync.py         # Synchronous FL server (FedAvg)
│   └── server_async.py        # Asynchronous FL server (NOVELTY)
│
├── optimization/
│   ├── client_selection.py    # Adaptive client selection
│   ├── energy.py              # Energy-aware training
│   └── compression.py         # Model compression
│
├── privacy/
│   └── differential_privacy.py # Device-level DP
│
├── aws_iot/
│   ├── iot_client.py          # AWS IoT MQTT client
│   └── iot_server.py          # AWS IoT MQTT server
│
├── evaluation/
│   └── metrics.py             # Evaluation and visualization
│
├── certificates/              # AWS IoT certificates (user-provided)
├── results/                   # Experiment results
├── run_experiments.py         # Main experiment runner
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Federated_IoT_Major_Project
pip install -r requirements.txt
```

### 2. Run All Experiments

```bash
python run_experiments.py
```

### 3. Run with Custom Options

```bash
# Run only async FL with DP enabled
python run_experiments.py --skip-centralized --skip-sync --enable-dp --num-rounds 50

# Run with compression
python run_experiments.py --enable-compression --clients-per-round 15

# Run without plots (for servers without display)
python run_experiments.py --no-plot
```

## 🔬 Novelty Modules

### 1. Asynchronous Federated Learning
- **File**: `federated/server_async.py`
- Server doesn't wait for all clients
- Weighted moving average aggregation
- Staleness-aware weight updates: `α = base_α × discount^staleness`
- Handles straggler devices gracefully

### 2. Adaptive Client Selection
- **File**: `optimization/client_selection.py`
- Considers battery level and network latency
- Selection Score = `battery_weight × battery + latency_weight × (1 - latency/max) + data_weight × data_size`
- Combines exploitation (best clients) with exploration
- Improves training efficiency on heterogeneous devices

### 3. Energy-Aware Training
- **File**: `optimization/energy.py`
- Dynamically adjusts local epochs based on battery:
  - Battery > 80%: Full epochs
  - 50-80%: epochs - 1
  - 30-50%: epochs - 2
  - < 30%: Minimum epochs
- Simulates realistic IoT energy constraints
- Extends device lifetime while maintaining accuracy

### 4. Communication Compression
- **File**: `optimization/compression.py`
- **Top-K sparsification**: Send only top K% of weights by magnitude
- **Random sparsification**: Randomly sample weights with scaling
- **Quantization**: 8-bit or 16-bit weight quantization
- Reduces bandwidth by up to 50%

### 5. Differential Privacy
- **File**: `privacy/differential_privacy.py`
- Device-level (ε, δ)-DP
- Gradient clipping to bound sensitivity
- Calibrated Gaussian noise: `σ = √(2 ln(1.25/δ)) × Δf / ε`
- Privacy-accuracy tradeoff analysis

## ☁️ AWS IoT Core Setup

### Prerequisites
1. AWS Account
2. AWS IoT Core enabled
3. Device certificates generated

### Configuration

1. Create an IoT Thing in AWS Console
2. Download certificates:
   - Device certificate (`*.pem.crt`)
   - Private key (`*-private.pem.key`)
   - Amazon Root CA (`AmazonRootCA1.pem`)

3. Place certificates in the `certificates/` folder

4. Update paths in `aws_iot/iot_client.py`:
```python
# TODO: Set your AWS IoT Core endpoint
self.endpoint = "YOUR_IOT_ENDPOINT.iot.YOUR_REGION.amazonaws.com"

# TODO: Set paths to your certificates
self.cert_path = "./certificates/device.pem.crt"
self.key_path = "./certificates/private.pem.key"
self.ca_path = "./certificates/AmazonRootCA1.pem"
```

> **Note**: The framework works in simulation mode without AWS credentials for local testing.

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Classification accuracy on test set |
| Loss | Cross-entropy loss |
| Training Time | Total time for all rounds |
| Communication Cost | Total bytes transmitted |
| Energy Consumption | Simulated battery usage |
| Staleness | Average model staleness (async FL) |

## 📈 Expected Results

| Method | Accuracy | Training Time | Privacy |
|--------|----------|---------------|---------|
| Centralized | ~95% | Baseline | ❌ No |
| Sync FL | ~92% | ~1.2x | ✅ Yes |
| Async FL | ~91% | ~0.8x | ✅ Yes |
| Async FL + DP | ~88% | ~0.8x | ✅✅ Strong |

## 🎓 Academic Context

This project aligns with recent IEEE publications on:
- Federated Learning for IoT (IEEE IoT Journal 2024)
- Asynchronous FL with staleness handling
- Privacy-preserving distributed learning
- Resource-efficient edge computing

### Suitable For
- Final Year B.Tech/M.Tech Major Project
- Research in Distributed ML
- Industry applications in Healthcare IoT, Smart Home, etc.

## 📝 Viva Preparation Points

### 1. Why Federated Learning?
- **Data privacy preservation**: Data never leaves the device
- **Reduced communication overhead**: Only model updates transmitted
- **No centralized data storage**: GDPR compliant

### 2. Why Asynchronous FL?
- **Handles device heterogeneity**: Different devices complete at different times
- **Reduces waiting time**: No need to wait for slowest device
- **Better for real-world IoT scenarios**: Devices may go offline

### 3. How does Differential Privacy work?
- **Adds calibrated noise**: Gaussian noise with std = σ × max_grad_norm
- **Provides mathematical privacy guarantees**: (ε, δ)-DP
- **Controlled by epsilon**: Lower ε = more privacy, less accuracy

### 4. AWS IoT Core role?
- **Secure MQTT communication**: TLS encryption
- **Device authentication**: X.509 certificates
- **Scalable message routing**: Pub/Sub model
- **Not for training**: Only for communication

### 5. FedAvg Algorithm
```
For each round t:
    1. Server selects subset S of clients
    2. Server sends global model W_t to selected clients
    3. Each client k trains locally: W_k = LocalTrain(W_t, D_k)
    4. Server aggregates: W_{t+1} = Σ (n_k/n) × W_k
```

### 6. Non-IID Data Distribution
- Each client has data from one user only
- Users have different activity patterns
- Some users walk more, others sit more
- Creates realistic heterogeneous scenario

## 🔧 Troubleshooting

### Dataset Download Issues
```bash
# Manual download
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip -d data/har_dataset/
```

### CUDA Out of Memory
```bash
# Use smaller batch size
python run_experiments.py --batch-size 16
```

### AWS IoT Connection Failed
- Check certificate paths
- Verify IoT endpoint URL
- Ensure Thing policy allows publish/subscribe

### Import Errors
```bash
# Make sure you're in the project directory
cd Federated_IoT_Major_Project
python run_experiments.py
```

## 📜 License

This project is for educational purposes. Dataset is from UCI ML Repository.

## 👥 Contributors

- **Student Name**: [Your Name]
- **Project Guide**: [Guide Name]
- **Institution**: [Your Institution]

---

## 🎯 Key Novelties Summary

1. **Asynchronous FL with Staleness-Aware Aggregation**
2. **Adaptive Battery and Latency-Aware Client Selection**
3. **Energy-Aware Dynamic Epoch Scheduling**
4. **Communication-Efficient Model Compression**
5. **Device-Level Differential Privacy**
6. **AWS IoT Core Integration for Secure Communication**

---

**Built with ❤️ for Final Year Major Project**

*Last Updated: January 2026*
