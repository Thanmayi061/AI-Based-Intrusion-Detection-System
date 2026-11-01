from collections import defaultdict


class FlowTracker:

    def __init__(self):
        self.flows = defaultdict(list)

    def get_flow_key(self, features):

        return (
            features["Source IP"],
            features["Destination IP"],
            features["Source Port"],
            features["Destination Port"],
            features["Protocol"]
        )

    def add_packet(self, features):

        key = self.get_flow_key(features)

        self.flows[key].append(features)

    def get_all_flows(self):

        return self.flows