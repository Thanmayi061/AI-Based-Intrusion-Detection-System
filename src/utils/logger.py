import csv
import os
from datetime import datetime

# ============================================
# Log File Path
# ============================================

LOG_FOLDER = "logs"
LOG_FILE = os.path.join(LOG_FOLDER, "predictions.csv")

# Create logs folder if it doesn't exist
os.makedirs(LOG_FOLDER, exist_ok=True)


# ============================================
# Save Prediction
# ============================================

def log_prediction(src_ip, dst_ip, protocol, prediction, confidence):

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Source IP",
                "Destination IP",
                "Protocol",
                "Prediction",
                "Confidence (%)"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            src_ip,
            dst_ip,
            protocol,
            prediction,
            f"{confidence:.2f}"
        ])