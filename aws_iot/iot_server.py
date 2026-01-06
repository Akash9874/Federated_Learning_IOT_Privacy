"""
AWS IoT Core Server
===================
MQTT-based server for federated learning aggregation.
Subscribes to model updates and broadcasts global model.

NOTE: This requires AWS IoT Core credentials and certificates.
"""

import json
import time
import threading
from queue import Queue
from typing import Dict, List, Callable, Optional
import torch
from collections import OrderedDict
import base64
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# AWS IoT SDK import
try:
    from awsiot import mqtt_connection_builder
    from awscrt import mqtt
    AWS_IOT_AVAILABLE = True
except ImportError:
    AWS_IOT_AVAILABLE = False


class AWSIoTServer:
    """
    AWS IoT Core MQTT Server for Federated Learning.
    
    Handles:
    - Subscribing to model updates from clients
    - Broadcasting global model to all clients
    - Managing client connections
    
    IMPORTANT: This is the COMMUNICATION layer only.
    Model aggregation logic is in server_sync.py and server_async.py.
    """
    
    # MQTT Topics (same as client)
    TOPIC_MODEL_UPDATE = "federated/model/update"
    TOPIC_GLOBAL_MODEL = "federated/model/global"
    TOPIC_CLIENT_STATUS = "federated/client/status"
    TOPIC_SERVER_COMMAND = "federated/server/command"
    
    def __init__(
        self,
        server_id: str = "fl_server",
        endpoint: str = None,
        cert_path: str = None,
        key_path: str = None,
        ca_path: str = None
    ):
        """
        Initialize AWS IoT server.
        
        Args:
            server_id: Unique identifier for the server
            endpoint: AWS IoT endpoint
            cert_path: Path to device certificate
            key_path: Path to private key
            ca_path: Path to Amazon Root CA certificate
            
        TODO: Replace these with your actual AWS IoT credentials
        """
        self.server_id = server_id
        
        # AWS IoT Configuration
        # TODO: Set your AWS IoT Core endpoint
        self.endpoint = endpoint or "YOUR_IOT_ENDPOINT.iot.YOUR_REGION.amazonaws.com"
        
        # TODO: Set paths to server certificates
        self.cert_path = cert_path or "./certificates/server.pem.crt"
        self.key_path = key_path or "./certificates/server_private.pem.key"
        self.ca_path = ca_path or "./certificates/AmazonRootCA1.pem"
        
        # Connection state
        self.connected = False
        self.mqtt_connection = None
        
        # Model update queue
        self.update_queue = Queue()
        self.client_status: Dict[str, Dict] = {}
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        
        # Callbacks
        self.on_model_update: Optional[Callable] = None
        
    def connect(self) -> bool:
        """
        Establish MQTT connection to AWS IoT Core.
        
        Returns:
            True if connection successful
        """
        if not AWS_IOT_AVAILABLE:
            print("AWS IoT SDK not available. Using simulation mode.")
            self.connected = True
            return True
        
        try:
            # Build MQTT connection
            self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.endpoint,
                cert_filepath=self.cert_path,
                pri_key_filepath=self.key_path,
                ca_filepath=self.ca_path,
                client_id=self.server_id,
                clean_session=False,
                keep_alive_secs=30
            )
            
            # Connect
            connect_future = self.mqtt_connection.connect()
            connect_future.result(timeout=10)
            
            self.connected = True
            print(f"Server {self.server_id} connected to AWS IoT Core")
            
            # Subscribe to topics
            self._subscribe_to_topics()
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to AWS IoT Core: {e}")
            print("Running in SIMULATION mode.")
            self.connected = True
            return True
    
    def _subscribe_to_topics(self) -> None:
        """Subscribe to client topics."""
        if self.mqtt_connection is None:
            return
        
        # Subscribe to model updates
        subscribe_future, _ = self.mqtt_connection.subscribe(
            topic=self.TOPIC_MODEL_UPDATE,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_model_update_received
        )
        subscribe_future.result()
        
        # Subscribe to client status
        subscribe_future, _ = self.mqtt_connection.subscribe(
            topic=self.TOPIC_CLIENT_STATUS,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_status_received
        )
        subscribe_future.result()
        
        print(f"Server subscribed to topics")
    
    def _on_model_update_received(
        self, 
        topic: str, 
        payload: bytes, 
        **kwargs
    ) -> None:
        """Handle received model updates from clients."""
        try:
            message = json.loads(payload.decode('utf-8'))
            
            client_id = message.get('client_id')
            weights = self._deserialize_weights(message.get('weights'))
            stats = message.get('stats', {})
            
            update = {
                'client_id': client_id,
                'weights': weights,
                'stats': stats,
                'timestamp': time.time()
            }
            
            self.update_queue.put(update)
            
            if self.on_model_update:
                self.on_model_update(update)
                
            print(f"Received model update from client {client_id}")
            
        except Exception as e:
            print(f"Error processing model update: {e}")
    
    def _on_status_received(
        self, 
        topic: str, 
        payload: bytes, 
        **kwargs
    ) -> None:
        """Handle client status updates."""
        try:
            message = json.loads(payload.decode('utf-8'))
            client_id = message.get('client_id')
            status = message.get('status', {})
            
            with self.lock:
                self.client_status[client_id] = {
                    **status,
                    'last_seen': time.time()
                }
                
        except Exception as e:
            print(f"Error processing status update: {e}")
    
    def broadcast_global_model(self, global_weights: OrderedDict) -> bool:
        """
        Broadcast global model to all clients.
        
        Args:
            global_weights: Global model weights to broadcast
            
        Returns:
            True if broadcast successful
        """
        try:
            weights_serialized = self._serialize_weights(global_weights)
            
            message = {
                'server_id': self.server_id,
                'timestamp': time.time(),
                'weights': weights_serialized,
                'type': 'global_model'
            }
            
            if self.mqtt_connection is not None:
                self.mqtt_connection.publish(
                    topic=self.TOPIC_GLOBAL_MODEL,
                    payload=json.dumps(message),
                    qos=mqtt.QoS.AT_LEAST_ONCE
                )
            else:
                print(f"[SIMULATED] Server broadcast global model")
            
            return True
            
        except Exception as e:
            print(f"Failed to broadcast global model: {e}")
            return False
    
    def send_command(self, command: Dict) -> bool:
        """
        Send command to clients.
        
        Args:
            command: Command dictionary (e.g., start_training, stop, etc.)
            
        Returns:
            True if sent successfully
        """
        try:
            message = {
                'server_id': self.server_id,
                'timestamp': time.time(),
                'command': command
            }
            
            if self.mqtt_connection is not None:
                self.mqtt_connection.publish(
                    topic=self.TOPIC_SERVER_COMMAND,
                    payload=json.dumps(message),
                    qos=mqtt.QoS.AT_LEAST_ONCE
                )
            
            return True
            
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False
    
    def get_pending_updates(self, timeout: float = 0.1) -> List[Dict]:
        """
        Get all pending model updates from the queue.
        
        Args:
            timeout: Timeout for each queue check
            
        Returns:
            List of pending updates
        """
        updates = []
        
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get(timeout=timeout)
                updates.append(update)
            except:
                break
        
        return updates
    
    def get_active_clients(self, timeout: float = 60.0) -> List[str]:
        """
        Get list of recently active clients.
        
        Args:
            timeout: Consider clients inactive after this many seconds
            
        Returns:
            List of active client IDs
        """
        current_time = time.time()
        active = []
        
        with self.lock:
            for client_id, status in self.client_status.items():
                last_seen = status.get('last_seen', 0)
                if current_time - last_seen < timeout:
                    active.append(client_id)
        
        return active
    
    def _serialize_weights(self, weights: OrderedDict) -> str:
        """Serialize model weights to base64 string."""
        numpy_weights = {k: v.cpu().numpy() for k, v in weights.items()}
        pickled = pickle.dumps(numpy_weights)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return encoded
    
    def _deserialize_weights(self, encoded: str) -> OrderedDict:
        """Deserialize model weights from base64 string."""
        decoded = base64.b64decode(encoded.encode('utf-8'))
        numpy_weights = pickle.loads(decoded)
        tensor_weights = OrderedDict({k: torch.from_numpy(v) for k, v in numpy_weights.items()})
        return tensor_weights
    
    def disconnect(self) -> None:
        """Disconnect from AWS IoT Core."""
        self.running = False
        if self.mqtt_connection is not None:
            self.mqtt_connection.disconnect()
        self.connected = False
        print(f"Server {self.server_id} disconnected")


class IoTFederatedLearningOrchestrator:
    """
    Orchestrates federated learning using AWS IoT Core.
    
    Combines the IoT server with FL aggregation logic.
    """
    
    def __init__(
        self,
        server: AWSIoTServer,
        aggregation_strategy: str = 'fedavg'
    ):
        """
        Initialize the orchestrator.
        
        Args:
            server: AWSIoTServer instance
            aggregation_strategy: 'fedavg' or 'async'
        """
        self.server = server
        self.aggregation_strategy = aggregation_strategy
        self.global_weights: Optional[OrderedDict] = None
        self.round_number = 0
    
    def run_round(
        self,
        min_clients: int = 3,
        timeout: float = 60.0
    ) -> Dict:
        """
        Run one round of federated learning over IoT.
        
        Args:
            min_clients: Minimum clients needed
            timeout: Timeout waiting for updates
            
        Returns:
            Round statistics
        """
        self.round_number += 1
        
        # Broadcast global model
        if self.global_weights is not None:
            self.server.broadcast_global_model(self.global_weights)
        
        # Send training command
        self.server.send_command({
            'action': 'train',
            'round': self.round_number
        })
        
        # Wait for updates
        start_time = time.time()
        updates = []
        
        while time.time() - start_time < timeout:
            new_updates = self.server.get_pending_updates()
            updates.extend(new_updates)
            
            if len(updates) >= min_clients:
                break
            
            time.sleep(1.0)
        
        if len(updates) == 0:
            return {'status': 'failed', 'reason': 'no_updates'}
        
        # Aggregate updates
        weights_list = [u['weights'] for u in updates]
        sample_counts = [u['stats'].get('train_samples', 1) for u in updates]
        
        self.global_weights = self._aggregate(weights_list, sample_counts)
        
        return {
            'status': 'completed',
            'round': self.round_number,
            'num_updates': len(updates),
            'clients': [u['client_id'] for u in updates]
        }
    
    def _aggregate(
        self, 
        weights_list: List[OrderedDict],
        sample_counts: List[int]
    ) -> OrderedDict:
        """Aggregate model weights using FedAvg."""
        total_samples = sum(sample_counts)
        
        averaged = OrderedDict()
        for key in weights_list[0].keys():
            averaged[key] = torch.zeros_like(weights_list[0][key], dtype=torch.float32)
            
            for weights, count in zip(weights_list, sample_counts):
                averaged[key] += weights[key].float() * (count / total_samples)
        
        return averaged


if __name__ == "__main__":
    # Test IoT server
    print("AWS IoT Server Test")
    print("="*50)
    
    server = AWSIoTServer()
    server.connect()
    
    # Test broadcasting
    test_weights = OrderedDict({
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10)
    })
    
    server.broadcast_global_model(test_weights)
    server.send_command({'action': 'start_training', 'round': 1})
    
    # Disconnect
    server.disconnect()
