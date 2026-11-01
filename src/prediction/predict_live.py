import warnings
import joblib
import pandas as pd

# Hide FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# Load AI Model
# ==========================================

model = joblib.load("models/live_random_forest_model.pkl")
scaler = joblib.load("models/live_scaler.pkl")

# ==========================================
# Attack Labels
# ==========================================

LABELS = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS Slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack - Brute Force",
    13: "Web Attack - SQL Injection",
    14: "Web Attack - XSS"
}


# ==========================================
# Live Prediction
# ==========================================

def predict_live(flow_features):

    # Create DataFrame
    data = pd.DataFrame([flow_features])

    # Keep feature order same as training
    data = data[
        [
            "Protocol",
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Packet Length Min",
            "Packet Length Max",
            "Avg Packet Size"
        ]
    ]

    # Convert every column to numeric
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Replace missing values with 0
    data = data.fillna(0)

    # Convert to float
    data = data.astype(float)

    # Scale features
    data_scaled = scaler.transform(data)

    # Predict class
    prediction = int(model.predict(data_scaled)[0])

    # Prediction probability
    probability = model.predict_proba(data_scaled)

    confidence = probability.max() * 100

    # Attack name
    attack = LABELS.get(prediction, f"Unknown ({prediction})")

    return attack, confidence