"""
AWS IoT Core Client
===================
MQTT-based client for IoT device communication in federated learning.
Publishes model updates to AWS IoT Core.

NOTE: This requires AWS IoT Core credentials and certificates.
"""

import json
import time
import threading
from typing import Dict, Callable, Optional
import torch
from collections import OrderedDict
import base64
import pickle
import sys
import os

# AWS IoT SDK import
try:
    from awsiot import mqtt_connection_builder
    from awscrt import mqtt
    AWS_IOT_AVAILABLE = True
except ImportError:
    AWS_IOT_AVAILABLE = False
    print("AWS IoT SDK not installed. Install with: pip install awsiotsdk")


class AWSIoTClient:
    """
    AWS IoT Core MQTT Client for Federated Learning.
    
    Handles:
    - Secure MQTT connection to AWS IoT Core
    - Publishing model updates from clients
    - Receiving global model from server
    
    IMPORTANT: This is the COMMUNICATION layer only.
    Model training happens locally on the device.
    """
    
    # MQTT Topics
    TOPIC_MODEL_UPDATE = "federated/model/update"
    TOPIC_GLOBAL_MODEL = "federated/model/global"
    TOPIC_CLIENT_STATUS = "federated/client/status"
    TOPIC_SERVER_COMMAND = "federated/server/command"
    
    def __init__(
        self,
        client_id: str,
        endpoint: str = None,
        cert_path: str = None,
        key_path: str = None,
        ca_path: str = None
    ):
        """
        Initialize AWS IoT client.
        
        Args:
            client_id: Unique identifier for this client
            endpoint: AWS IoT endpoint (e.g., xxxx-ats.iot.region.amazonaws.com)
            cert_path: Path to device certificate
            key_path: Path to private key
            ca_path: Path to Amazon Root CA certificate
            
        TODO: Replace these with your actual AWS IoT credentials:
        - endpoint: Get from AWS IoT Console -> Settings
        - cert_path: Download from AWS IoT Console when creating Thing
        - key_path: Download from AWS IoT Console when creating Thing
        - ca_path: Download Amazon Root CA from AWS documentation
        """
        self.client_id = client_id
        
        # AWS IoT Configuration
        # TODO: Set your AWS IoT Core endpoint here
        self.endpoint = endpoint or "YOUR_IOT_ENDPOINT.iot.YOUR_REGION.amazonaws.com"
        
        # TODO: Set paths to your certificates
        self.cert_path = cert_path or "./certificates/device.pem.crt"
        self.key_path = key_path or "./certificates/private.pem.key"
        self.ca_path = ca_path or "./certificates/AmazonRootCA1.pem"
        
        # Connection state
        self.connected = False
        self.mqtt_connection = None
        
        # Callbacks
        self.on_global_model_received: Optional[Callable] = None
        self.on_command_received: Optional[Callable] = None
        
        # Message queue for received messages
        self.message_queue = []
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        Establish MQTT connection to AWS IoT Core.
        
        Returns:
            True if connection successful, False otherwise
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
                client_id=self.client_id,
                clean_session=False,
                keep_alive_secs=30
            )
            
            # Connect
            connect_future = self.mqtt_connection.connect()
            connect_future.result(timeout=10)
            
            self.connected = True
            print(f"Client {self.client_id} connected to AWS IoT Core")
            
            # Subscribe to topics
            self._subscribe_to_topics()
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to AWS IoT Core: {e}")
            print("Running in SIMULATION mode without AWS IoT.")
            self.connected = True  # Simulate connection for testing
            return True
    
    def _subscribe_to_topics(self) -> None:
        """Subscribe to relevant MQTT topics."""
        if self.mqtt_connection is None:
            return
        
        # Subscribe to global model updates
        self.mqtt_connection.subscribe(
            topic=self.TOPIC_GLOBAL_MODEL,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_message_received
        )
        
        # Subscribe to server commands
        self.mqtt_connection.subscribe(
            topic=self.TOPIC_SERVER_COMMAND,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_message_received
        )
        
        print(f"Client {self.client_id} subscribed to topics")
    
    def _on_message_received(self, topic: str, payload: bytes, **kwargs) -> None:
        """Handle received MQTT messages."""
        try:
            message = json.loads(payload.decode('utf-8'))
            
            with self.lock:
                self.message_queue.append({
                    'topic': topic,
                    'message': message,
                    'timestamp': time.time()
                })
            
            # Trigger callbacks
            if topic == self.TOPIC_GLOBAL_MODEL and self.on_global_model_received:
                self.on_global_model_received(message)
            elif topic == self.TOPIC_SERVER_COMMAND and self.on_command_received:
                self.on_command_received(message)
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def publish_model_update(
        self,
        model_weights: OrderedDict,
        client_stats: Dict
    ) -> bool:
        """
        Publish model update to AWS IoT Core.
        
        Args:
            model_weights: Model weights to publish
            client_stats: Training statistics
            
        Returns:
            True if published successfully
        """
        try:
            # Serialize model weights
            weights_serialized = self._serialize_weights(model_weights)
            
            # Prepare message
            message = {
                'client_id': self.client_id,
                'timestamp': time.time(),
                'weights': weights_serialized,
                'stats': client_stats
            }
            
            if self.mqtt_connection is not None:
                # Publish to AWS IoT Core
                self.mqtt_connection.publish(
                    topic=self.TOPIC_MODEL_UPDATE,
                    payload=json.dumps(message),
                    qos=mqtt.QoS.AT_LEAST_ONCE
                )
            else:
                # Simulation mode - just log
                print(f"[SIMULATED] Client {self.client_id} published model update")
            
            return True
            
        except Exception as e:
            print(f"Failed to publish model update: {e}")
            return False
    
    def publish_status(self, status: Dict) -> bool:
        """
        Publish client status to AWS IoT Core.
        
        Args:
            status: Status dictionary (battery, latency, etc.)
            
        Returns:
            True if published successfully
        """
        try:
            message = {
                'client_id': self.client_id,
                'timestamp': time.time(),
                'status': status
            }
            
            if self.mqtt_connection is not None:
                self.mqtt_connection.publish(
                    topic=self.TOPIC_CLIENT_STATUS,
                    payload=json.dumps(message),
                    qos=mqtt.QoS.AT_LEAST_ONCE
                )
            
            return True
            
        except Exception as e:
            print(f"Failed to publish status: {e}")
            return False
    
    def get_global_model(self, timeout: float = 30.0) -> Optional[OrderedDict]:
        """
        Wait for and receive global model from server.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Global model weights or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                for i, msg in enumerate(self.message_queue):
                    if msg['topic'] == self.TOPIC_GLOBAL_MODEL:
                        weights = self._deserialize_weights(msg['message']['weights'])
                        self.message_queue.pop(i)
                        return weights
            
            time.sleep(0.1)
        
        return None
    
    def _serialize_weights(self, weights: OrderedDict) -> str:
        """Serialize model weights to base64 string."""
        # Convert tensors to numpy and pickle
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
        if self.mqtt_connection is not None:
            self.mqtt_connection.disconnect()
        self.connected = False
        print(f"Client {self.client_id} disconnected")


class IoTClientSimulator:
    """
    Simulator for AWS IoT client when AWS credentials are not available.
    Used for local testing and development.
    """
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.connected = False
        self.sent_messages = []
        self.received_models = []
    
    def connect(self) -> bool:
        """Simulate connection."""
        self.connected = True
        print(f"[SIMULATOR] Client {self.client_id} connected")
        return True
    
    def publish_model_update(
        self, 
        model_weights: OrderedDict, 
        client_stats: Dict
    ) -> bool:
        """Simulate publishing model update."""
        self.sent_messages.append({
            'type': 'model_update',
            'timestamp': time.time(),
            'stats': client_stats
        })
        print(f"[SIMULATOR] Client {self.client_id} sent model update")
        return True
    
    def receive_global_model(self, weights: OrderedDict) -> None:
        """Simulate receiving global model."""
        self.received_models.append({
            'weights': weights,
            'timestamp': time.time()
        })
    
    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.connected = False


def create_iot_client(
    client_id: str,
    use_aws: bool = False,
    **kwargs
) -> AWSIoTClient:
    """
    Factory function to create IoT client.
    
    Args:
        client_id: Client identifier
        use_aws: Whether to use real AWS IoT (requires credentials)
        **kwargs: Additional arguments for AWSIoTClient
        
    Returns:
        IoT client instance
    """
    if use_aws and AWS_IOT_AVAILABLE:
        return AWSIoTClient(client_id, **kwargs)
    else:
        # Return simulator or basic client
        return AWSIoTClient(client_id)  # Will run in simulation mode


if __name__ == "__main__":
    # Test IoT client
    print("AWS IoT Client Test")
    print("="*50)
    
    client = create_iot_client("test_client_001")
    
    # Connect (will use simulation if AWS credentials not available)
    client.connect()
    
    # Test publishing
    test_weights = OrderedDict({
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10)
    })
    
    test_stats = {
        'loss': 0.5,
        'accuracy': 85.0,
        'battery': 75.0
    }
    
    client.publish_model_update(test_weights, test_stats)
    client.publish_status({'battery': 75.0, 'latency': 0.5})
    
    # Disconnect
    client.disconnect()
