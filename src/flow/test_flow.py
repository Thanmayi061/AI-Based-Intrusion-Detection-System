from src.flow.flow_tracker import FlowTracker
from src.flow.flow_features import calculate_flow_features

tracker = FlowTracker()

packet1 = {
    "Source IP": "192.168.1.36",
    "Destination IP": "20.184.175.10",
    "Source Port": 50314,
    "Destination Port": 443,
    "Protocol": "TCP",
    "Packet Length": 571
}

packet2 = {
    "Source IP": "192.168.1.36",
    "Destination IP": "20.184.175.10",
    "Source Port": 50314,
    "Destination Port": 443,
    "Protocol": "TCP",
    "Packet Length": 54
}

packet3 = {
    "Source IP": "192.168.1.36",
    "Destination IP": "20.184.175.10",
    "Source Port": 50314,
    "Destination Port": 443,
    "Protocol": "TCP",
    "Packet Length": 78
}

tracker.add_packet(packet1)
tracker.add_packet(packet2)
tracker.add_packet(packet3)

flows = tracker.get_all_flows()

for flow_key, packets in flows.items():
    print("\nFlow:", flow_key)

    features = calculate_flow_features(packets)

    print(features)