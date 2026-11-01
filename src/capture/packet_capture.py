import os
import csv
from scapy.all import sniff

from src.utils.logger import log_prediction
from src.feature_extraction.feature_extractor import extract_features
from src.flow.flow_tracker import FlowTracker
from src.flow.flow_features import calculate_flow_features
from src.prediction.predict_live import predict_live

# ============================================
# Initialize Flow Tracker
# ============================================

tracker = FlowTracker()

# ============================================
# Output CSV
# ============================================

OUTPUT_FOLDER = "output"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "captured_packets.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Source IP",
            "Destination IP",
            "Protocol",
            "Protocol Number",
            "Source Port",
            "Destination Port",
            "Packet Length",
            "TTL",
            "TCP Flags"
        ])

# ============================================
# Process Packet
# ============================================

def process_packet(packet):

    # Extract packet features
    features = extract_features(packet)

    # Print packet information
    print("\n" + "=" * 60)
    print("Packet Features")
    print("=" * 60)

    for key, value in features.items():
        print(f"{key:<20}: {value}")

    # Save packet to CSV
    with open(OUTPUT_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            features["Source IP"],
            features["Destination IP"],
            features["Protocol"],
            features["Protocol Number"],
            features["Source Port"],
            features["Destination Port"],
            features["Packet Length"],
            features["TTL"],
            features["TCP Flags"]
        ])

    # ============================================
    # Flow Tracking
    # ============================================

    tracker.add_packet(features)

    flow_key = tracker.get_flow_key(features)

    packet_list = tracker.get_all_flows()[flow_key]

    flow_stats = calculate_flow_features(packet_list)

    print("\n" + "-" * 60)
    print("Flow Statistics")
    print("-" * 60)

    for key, value in flow_stats.items():
        print(f"{key:<25}: {value}")

    # ============================================
    # AI Prediction
    # ============================================

    try:

        live_features = {
            "Protocol": features["Protocol Number"],
            "Flow Duration": flow_stats["Flow Duration"],
            "Flow Bytes/s": flow_stats["Bytes per Second"],
            "Flow Packets/s": flow_stats["Packets per Second"],
            "Packet Length Min": flow_stats["Minimum Packet Size"],
            "Packet Length Max": flow_stats["Maximum Packet Size"],
            "Avg Packet Size": flow_stats["Average Packet Size"]
        }

        # AI Prediction
        prediction, confidence = predict_live(live_features)

        # Save prediction log
        log_prediction(
            features["Source IP"],
            features["Destination IP"],
            features["Protocol"],
            prediction,
            confidence
        )

        # Display prediction
        print("\n" + "=" * 60)
        print("AI PREDICTION")
        print("=" * 60)

        if prediction == "BENIGN":
            print("✅ SAFE NETWORK TRAFFIC")
        else:
            print("⚠ ATTACK DETECTED")

        print("Prediction :", prediction)
        print(f"Confidence : {confidence:.2f}%")

    except Exception as e:

        print("\n" + "=" * 60)
        print("AI PREDICTION ERROR")
        print("=" * 60)
        print(e)

# ============================================
# Start Packet Capture
# ============================================

def start_capture():

    print("=" * 60)
    print(" SMART AI INTRUSION DETECTION SYSTEM ")
    print("=" * 60)
    print("Capturing live packets...")
    print("Press Ctrl + C to stop.")
    print("=" * 60)

    sniff(
        prn=process_packet,
        store=False
    )

# ============================================
# Main
# ============================================

if __name__ == "__main__":
    start_capture()