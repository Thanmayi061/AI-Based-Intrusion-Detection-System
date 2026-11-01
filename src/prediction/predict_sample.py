from pathlib import Path
import pandas as pd

from predictor import predict_attack


# Load one sample from the processed test data
X_test = pd.read_parquet(
    Path("data/processed") / "X_test.parquet"
)

sample = X_test.iloc[[0]]

prediction = predict_attack(sample)

print("\nPrediction Result")
print("------------------------")
print(prediction[0])