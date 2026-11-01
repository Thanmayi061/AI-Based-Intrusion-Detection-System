from pathlib import Path
import joblib
import pandas as pd
# PROJECT PATHS
PROCESSED_FOLDER = Path("data/processed")
MODELS_FOLDER = Path("models")
def load_processed_data():
    print("\nLoading processed datasets...\n")
    X_train = pd.read_parquet(PROCESSED_FOLDER / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_FOLDER / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_FOLDER / "y_train.parquet")
    y_test = pd.read_parquet(PROCESSED_FOLDER / "y_test.parquet")
    y_train = y_train["Label"]
    y_test = y_test["Label"]
    print("Processed datasets loaded successfully.")
    return X_train, X_test, y_train, y_test
def save_model(model, filename):
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = MODELS_FOLDER / filename
    joblib.dump(model, output_file)
    print(f"\nModel saved : {output_file}")