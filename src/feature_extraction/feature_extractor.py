import time

from scapy.layers.inet import IP, TCP, UDP


def extract_features(packet):
    """
    Extract important features from a network packet.
    """

    features = {}

    # ---------------------------------
    # IP Layer
    # ---------------------------------
    if packet.haslayer(IP):

        ip = packet[IP]

        features["Source IP"] = ip.src
        features["Destination IP"] = ip.dst
        features["TTL"] = ip.ttl
        features["Protocol Number"] = ip.proto

    else:

        features["Source IP"] = None
        features["Destination IP"] = None
        features["TTL"] = None
        features["Protocol Number"] = None

    # ---------------------------------
    # TCP Layer
    # ---------------------------------
    if packet.haslayer(TCP):

        tcp = packet[TCP]

        features["Protocol"] = "TCP"
        features["Source Port"] = tcp.sport
        features["Destination Port"] = tcp.dport
        features["TCP Flags"] = str(tcp.flags)

    # ---------------------------------
    # UDP Layer
    # ---------------------------------
    elif packet.haslayer(UDP):

        udp = packet[UDP]

        features["Protocol"] = "UDP"
        features["Source Port"] = udp.sport
        features["Destination Port"] = udp.dport
        features["TCP Flags"] = None

    # ---------------------------------
    # Other Protocols
    # ---------------------------------
    else:

        features["Protocol"] = "Other"
        features["Source Port"] = None
        features["Destination Port"] = None
        features["TCP Flags"] = None

    # ---------------------------------
    # Packet Information
    # ---------------------------------

    features["Packet Length"] = len(packet)

    # Capture Timestamp
    features["Timestamp"] = time.time()

    return features