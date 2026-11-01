import pandas as pd

# Load combined dataset
df = pd.read_parquet("data/processed/live_training_dataset.parquet")

# Features available from live capture
selected_features = [
    "Protocol",
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Avg Packet Size",
    "Label"
]

# Create reduced dataset
live_df = df[selected_features]

# Save
live_df.to_parquet(
    "data/processed/live_features_dataset.parquet",
    index=False
)

print("=" * 60)
print("Live Features Dataset Created Successfully")
print("=" * 60)
print(f"Rows    : {len(live_df)}")
print(f"Columns : {len(live_df.columns)}")
print("\nColumns:")
print(live_df.columns.tolist())