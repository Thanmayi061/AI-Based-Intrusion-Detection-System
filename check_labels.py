import pandas as pd

df = pd.read_parquet("data/processed/live_training_dataset.parquet")

print("Unique Labels:")
print(df["Label"].unique())

print("\nLabel Counts:")
print(df["Label"].value_counts().sort_index())