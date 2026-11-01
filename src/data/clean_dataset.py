# IMPORT LIBRARIES
from pathlib import Path
import pandas as pd
import numpy as np

# PROJECT PATHS
RAW_DATA_FOLDER = Path("data/raw")
CLEANED_DATA_FOLDER = Path("data/cleaned")

# Create cleaned folder if it doesn't exist
CLEANED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# LOAD DATASETS
def load_datasets():
    parquet_files = sorted(RAW_DATA_FOLDER.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            "No parquet datasets found in data/raw")
    return parquet_files

# REMOVE CONSTANT COLUMNS
def remove_constant_columns(df):
    constant_columns = []
    for column in df.columns:
        if column == "Label":
            continue
        if df[column].nunique(dropna=False) == 1:
            constant_columns.append(column)
    if constant_columns:
        df = df.drop(columns=constant_columns)
    return df, constant_columns

# REMOVE UNCOMMON COLUMNS
def remove_uncommon_columns(df):
    uncommon_columns = [
        "CWE Flag Count",
        "Fwd URG Flags"
    ]
    removed_columns = []
    for column in uncommon_columns:
        if column in df.columns:
            df = df.drop(columns=column)
            removed_columns.append(column)
    return df, removed_columns

# REMOVE DUPLICATE ROWS
def remove_duplicate_rows(df):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed

# HANDLE MISSING VALUES
def handle_missing_values(df):
    missing = df.isnull().sum().sum()
    return df, missing

# HANDLE INFINITE VALUES
def handle_infinite_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    infinite_count = np.isinf(df[numeric_columns]).sum().sum()
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan)
    return df, infinite_count

# CLEAN DATASET
def clean_dataset(files):
    print("\n" + "=" * 80)
    print("DATA CLEANING")
    print("=" * 80)
    for file in files:
        print(f"\nProcessing : {file.name}")
        df = pd.read_parquet(file)
        rows_before = df.shape[0]
        columns_before = df.shape[1]
        df, constant_columns = remove_constant_columns(df)
        #Remove uncommon columns
        df, uncommon_columns = remove_uncommon_columns(df)

         # Remove duplicate rows
        df, duplicate_rows = remove_duplicate_rows(df)

        # Missing values
        df, missing_values = handle_missing_values(df)

        # Infinite values
        df, infinite_values = handle_infinite_values(df)

        rows_after = df.shape[0]
        columns_after = df.shape[1]
        # Save cleaned dataset
        output_file = CLEANED_DATA_FOLDER / file.name
        df.to_parquet(output_file, index=False)

        # Display Summary
        print(f"Rows Before               : {rows_before}")
        print(f"Rows After                : {rows_after}")
        print(f"Columns Before            : {columns_before}")
        print(f"Columns After             : {columns_after}")
        print(f"Constant Columns Removed  : {len(constant_columns)}")
        print(f"Uncommon Columns Removed  : {len(uncommon_columns)}")
        print(f"Duplicate Rows Removed    : {duplicate_rows}")
        print(f"Missing Values            : {missing_values}")
        print(f"Infinite Values           : {infinite_values}")
        print(f"Saved File                : {output_file.name}")

# MAIN CONTROLLER
def clean():
    files = load_datasets()
    clean_dataset(files)