"""
AWS IoT Core Demo for Federated Learning
=========================================
This script demonstrates AWS IoT Core integration.
Run this to show your AWS IoT implementation working!

SETUP REQUIRED:
1. Place your certificates in the 'certificates/' folder
2. Update the ENDPOINT below with your AWS IoT endpoint
3. Run: python aws_iot_demo.py
"""

import json
import time
import os
import sys

# Fix for Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================
# AWS IoT CONFIGURATION
# ============================================
ENDPOINT = "awrt2ux3uxh2u-ats.iot.ap-southeast-2.amazonaws.com"
CLIENT_ID = "federated_client_demo"
CERT_PATH = "./certificates/device.crt.pem"
KEY_PATH = "./certificates/private.pem.key"
CA_PATH = "./certificates/AmazonRootCA1.pem"

# MQTT Topics
TOPIC_PUBLISH = "federated/model/update"
TOPIC_SUBSCRIBE = "federated/model/global"

# Try to import AWS IoT SDK
try:
    from awsiot import mqtt_connection_builder
    from awscrt import mqtt
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("AWS IoT SDK not found. Install with: pip install awsiotsdk")


def check_certificates():
    """Check if certificates exist."""
    print("\n" + "="*60)
    print("CHECKING CERTIFICATES")
    print("="*60)
    
    certs = [
        ("Device Certificate", CERT_PATH),
        ("Private Key", KEY_PATH),
        ("Root CA", CA_PATH)
    ]
    
    all_exist = True
    for name, path in certs:
        exists = os.path.exists(path)
        status = "[OK] Found" if exists else "[X] Missing"
        print(f"{name}: {status}")
        if not exists:
            all_exist = False
    
    return all_exist


def on_message_received(topic, payload, **kwargs):
    """Callback when message is received."""
    print(f"\n[RECEIVED] Message on topic: {topic}")
    try:
        message = json.loads(payload.decode('utf-8'))
        print(f"   Content: {json.dumps(message, indent=2)}")
    except:
        print(f"   Raw payload: {payload}")


def run_aws_iot_demo():
    """Run the AWS IoT demonstration."""
    
    print("\n" + "="*60)
    print("AWS IoT CORE DEMO - FEDERATED LEARNING")
    print("="*60)
    
    # Check prerequisites
    if not AWS_AVAILABLE:
        print("\n[ERROR] AWS IoT SDK not available!")
        print("Install with: pip install awsiotsdk")
        return False
    
    if ENDPOINT.startswith("YOUR_"):
        print("\n[ERROR] Please update the ENDPOINT in this file!")
        print("Get your endpoint from AWS IoT Console -> Settings")
        return False
    
    if not check_certificates():
        print("\n[ERROR] Missing certificates!")
        print("Download from AWS IoT Console when creating Thing")
        return False
    
    # Connect to AWS IoT Core
    print("\n" + "="*60)
    print("CONNECTING TO AWS IoT CORE")
    print("="*60)
    
    try:
        # Build MQTT connection
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            cert_filepath=CERT_PATH,
            pri_key_filepath=KEY_PATH,
            ca_filepath=CA_PATH,
            client_id=CLIENT_ID,
            clean_session=False,
            keep_alive_secs=30
        )
        
        print(f"Connecting to {ENDPOINT}...")
        connect_future = mqtt_connection.connect()
        connect_future.result(timeout=10)
        print("[SUCCESS] Connected to AWS IoT Core!")
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False
    
    # Subscribe to topic
    print("\n" + "="*60)
    print("SUBSCRIBING TO TOPIC")
    print("="*60)
    
    try:
        subscribe_future, _ = mqtt_connection.subscribe(
            topic=TOPIC_SUBSCRIBE,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message_received
        )
        subscribe_future.result(timeout=10)
        print(f"[SUCCESS] Subscribed to: {TOPIC_SUBSCRIBE}")
        
    except Exception as e:
        print(f"[ERROR] Subscribe failed: {e}")
    
    # Publish messages (simulating FL client)
    print("\n" + "="*60)
    print("PUBLISHING MODEL UPDATES (Simulating FL Client)")
    print("="*60)
    
    for i in range(3):
        # Simulate a model update message
        message = {
            "client_id": CLIENT_ID,
            "round": i + 1,
            "timestamp": time.time(),
            "model_update": {
                "layer1_weights_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
                "loss": 0.5 - (i * 0.1),
                "accuracy": 70 + (i * 5)
            },
            "device_status": {
                "battery_level": 85 - (i * 5),
                "latency_ms": 150
            }
        }
        
        try:
            mqtt_connection.publish(
                topic=TOPIC_PUBLISH,
                payload=json.dumps(message),
                qos=mqtt.QoS.AT_LEAST_ONCE
            )
            print(f"\n[PUBLISH] Round {i+1} Model Update:")
            print(f"   Topic: {TOPIC_PUBLISH}")
            print(f"   Loss: {message['model_update']['loss']:.2f}")
            print(f"   Accuracy: {message['model_update']['accuracy']}%")
            print(f"   Battery: {message['device_status']['battery_level']}%")
            
        except Exception as e:
            print(f"[ERROR] Publish failed: {e}")
        
        time.sleep(2)  # Wait between messages
    
    # Wait for any incoming messages
    print("\n" + "="*60)
    print("WAITING FOR INCOMING MESSAGES (5 seconds)")
    print("="*60)
    print("(In real FL, server would send global model here)")
    time.sleep(5)
    
    # Disconnect
    print("\n" + "="*60)
    print("DISCONNECTING")
    print("="*60)
    
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result(timeout=10)
    print("[SUCCESS] Disconnected from AWS IoT Core!")
    
    print("\n" + "="*60)
    print("AWS IoT DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return True


def run_simulation_demo():
    """Run a simulation demo if AWS is not configured."""
    
    print("\n" + "="*60)
    print("AWS IoT SIMULATION DEMO")
    print("(Running without real AWS connection)")
    print("="*60)
    
    print("\n[SIM] Simulating MQTT Connection...")
    time.sleep(1)
    print("[SUCCESS] Connected to simulated broker")
    
    print(f"\n[SIM] Subscribing to: {TOPIC_SUBSCRIBE}")
    time.sleep(0.5)
    print("[SUCCESS] Subscribed successfully")
    
    print("\n" + "-"*60)
    print("SIMULATING FEDERATED LEARNING COMMUNICATION")
    print("-"*60)
    
    for i in range(3):
        message = {
            "client_id": "simulated_client",
            "round": i + 1,
            "model_weights_hash": f"hash_{i+1}_abc123",
            "loss": 0.5 - (i * 0.1),
            "accuracy": 70 + (i * 5),
            "battery": 85 - (i * 5)
        }
        
        print(f"\n[PUBLISH] Round {i+1} - Publishing model update:")
        print(f"   Topic: {TOPIC_PUBLISH}")
        print(f"   Message: {json.dumps(message, indent=6)}")
        time.sleep(1)
    
    print("\n" + "-"*60)
    print("SIMULATING SERVER RESPONSE")
    print("-"*60)
    
    global_model = {
        "server_id": "fl_server",
        "round": 3,
        "global_weights_hash": "aggregated_hash_xyz789",
        "participating_clients": 10,
        "global_accuracy": 89.5
    }
    
    print(f"\n[RECEIVED] Global model from server:")
    print(f"   Topic: {TOPIC_SUBSCRIBE}")
    print(f"   Message: {json.dumps(global_model, indent=6)}")
    
    print("\n[SUCCESS] Simulation completed!")
    print("\n" + "="*60)
    print("To run with REAL AWS IoT Core:")
    print("1. Create AWS account (free tier)")
    print("2. Create IoT Thing in AWS Console")
    print("3. Download certificates to ./certificates/")
    print("4. Update ENDPOINT in this file")
    print("5. Run again!")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FEDERATED LEARNING - AWS IoT CORE DEMONSTRATION")
    print("="*60)
    
    # Check if we should run real demo or simulation
    certs_exist = all([
        os.path.exists(CERT_PATH),
        os.path.exists(KEY_PATH),
        os.path.exists(CA_PATH)
    ])
    
    endpoint_configured = not ENDPOINT.startswith("YOUR_")
    
    if certs_exist and endpoint_configured and AWS_AVAILABLE:
        print("\n[OK] AWS IoT configured - Running REAL demo")
        success = run_aws_iot_demo()
    else:
        print("\n[WARNING] AWS IoT not fully configured - Running SIMULATION")
        if not AWS_AVAILABLE:
            print("   Reason: awsiotsdk not installed")
        if not endpoint_configured:
            print("   Reason: Endpoint not configured")
        if not certs_exist:
            print("   Reason: Certificates not found")
        run_simulation_demo()
