import time


def calculate_flow_features(packet_list):

    if len(packet_list) == 0:
        return None

    # ------------------------------------
    # Basic Information
    # ------------------------------------

    total_packets = len(packet_list)

    packet_lengths = [
        packet["Packet Length"]
        for packet in packet_list
    ]

    total_bytes = sum(packet_lengths)

    average_packet_size = total_bytes / total_packets

    minimum_packet_size = min(packet_lengths)

    maximum_packet_size = max(packet_lengths)

    # ------------------------------------
    # Flow Duration
    # ------------------------------------

    first_time = packet_list[0].get("Timestamp", time.time())

    last_time = packet_list[-1].get("Timestamp", time.time())

    flow_duration = max(last_time - first_time, 0.001)

    # ------------------------------------
    # Traffic Rate
    # ------------------------------------

    packets_per_second = total_packets / flow_duration

    bytes_per_second = total_bytes / flow_duration

    # ------------------------------------
    # Return Features
    # ------------------------------------

    return {

        "Flow Duration": round(flow_duration, 3),

        "Total Packets": total_packets,

        "Total Bytes": total_bytes,

        "Average Packet Size": round(average_packet_size, 2),

        "Minimum Packet Size": minimum_packet_size,

        "Maximum Packet Size": maximum_packet_size,

        "Packets per Second": round(packets_per_second, 2),

        "Bytes per Second": round(bytes_per_second, 2)

    }