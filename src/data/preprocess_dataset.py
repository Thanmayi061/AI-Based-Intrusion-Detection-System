from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# PROJECT PATHS
CLEANED_FOLDER = Path("data/cleaned")
PROCESSED_FOLDER = Path("data/processed")
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# LOAD CLEANED DATASETS
def load_cleaned_datasets():
    parquet_files = sorted(CLEANED_FOLDER.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No cleaned datasets found.")
    dataframes = []
    print("\nLoading cleaned datasets...\n")
    for file in parquet_files:
        df = pd.read_parquet(file)
        dataframes.append(df)
        print(f"Loaded : {file.name}")
    return dataframes

# MERGE DATASETS
def merge_datasets(dataframes):
    print("\nMerging datasets...\n")
    merged_df = pd.concat(
        dataframes,
        ignore_index=True
    )
    return merged_df

# DATASET INFORMATION
def dataset_information(df):
    print("\n" + "="*80)
    print("MERGED DATASET INFORMATION")
    print("="*80)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print("\nClass Distribution\n")
    print(df["Label"].value_counts())

# SAVE DATASET
def save_dataset(df):
    output_file = PROCESSED_FOLDER / "merged_dataset.parquet"
    df.to_parquet(output_file, index=False)
    print("\nMerged dataset saved successfully.")
    print(output_file)

# NORMALIZE LABELS
def normalize_labels(df):
    label_mapping = {
        "Benign": "Benign",
        "Bot": "Bot",
        "DDoS": "DDoS",
        "DoS Hulk": "DoS Hulk",
        "DoS GoldenEye": "DoS GoldenEye",
        "DoS slowloris": "DoS Slowloris",
        "DoS Slowhttptest": "DoS SlowHTTPTest",
        "FTP-Patator": "FTP-Patator",
        "SSH-Patator": "SSH-Patator",
        "PortScan": "PortScan",
        "Web Attack � Brute Force": "Web Attack Brute Force",
        "Web Attack � XSS": "Web Attack XSS",
        "Web Attack � Sql Injection": "Web Attack SQL Injection",
        "Heartbleed": "Heartbleed",
        "Infiltration": "Infiltration"
    }

    df["Label"] = df["Label"].replace(label_mapping)
    return df

# ENCODE LABELS
def encode_labels(df):
    encoder = LabelEncoder()
    df["Label"] = encoder.fit_transform(df["Label"])
    print("\nEncoded Classes\n")
    for index, label in enumerate(encoder.classes_):
        print(f"{index} --> {label}")
    return df, encoder

# SPLIT FEATURES AND TARGET
def split_features_target(df):
    X = df.drop(columns=["Label"])
    y = df["Label"]
    print("\nFeature Matrix :", X.shape)
    print("Target Vector  :", y.shape)
    return X, y
# TRAIN TEST SPLIT

def train_test_split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print("\n" + "=" * 80)
    print("TRAIN TEST SPLIT")
    print("=" * 80)
    print(f"Training Features : {X_train.shape}")
    print(f"Testing Features  : {X_test.shape}")
    print(f"Training Labels   : {y_train.shape}")
    print(f"Testing Labels    : {y_test.shape}")
    return X_train, X_test, y_train, y_test

# FEATURE SCALING
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n" + "=" * 80)
    print("FEATURE SCALING")
    print("=" * 80)
    print(f"Training Features : {X_train_scaled.shape}")
    print(f"Testing Features  : {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler

# SAVE PROCESSED DATA
def save_processed_data(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    X_columns,
    scaler,
    encoder
):

    print("\nSaving processed datasets...\n")

    # Convert NumPy arrays back to DataFrames
    X_train_df = pd.DataFrame(
        X_train_scaled,
        columns=X_columns
    )

    X_test_df = pd.DataFrame(
        X_test_scaled,
        columns=X_columns
    )

    y_train_df = pd.DataFrame(y_train, columns=["Label"])

    y_test_df = pd.DataFrame(y_test, columns=["Label"])

    # Save datasets
    X_train_df.to_parquet(
        PROCESSED_FOLDER / "X_train.parquet",
        index=False
    )

    X_test_df.to_parquet(
        PROCESSED_FOLDER / "X_test.parquet",
        index=False
    )

    y_train_df.to_parquet(
        PROCESSED_FOLDER / "y_train.parquet",
        index=False
    )

    y_test_df.to_parquet(
        PROCESSED_FOLDER / "y_test.parquet",
        index=False
    )

    # Save scaler
    joblib.dump(
        scaler,
        Path("models") / "scaler.pkl"
    )

    # Save label encoder
    joblib.dump(
        encoder,
        Path("models") / "label_encoder.pkl"
    )
    print("Processed datasets saved successfully.")
# MAIN
def preprocess():
  # Load cleaned datasets
    datasets = load_cleaned_datasets()
    # Merge datasets
    merged_df = merge_datasets(datasets)
    # Display dataset information
    dataset_information(merged_df)
    # Normalize labels
    merged_df = normalize_labels(merged_df)
    # Save merged dataset with text labels
    save_dataset(merged_df)
    # Encode labels
    merged_df, encoder = encode_labels(merged_df)
    # Split features and target
    X, y = split_features_target(merged_df)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    # Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )
    save_processed_data(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    X.columns,
    scaler,
    encoder
)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)

