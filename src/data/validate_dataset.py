from pathlib import Path
import pandas as pd
import numpy as np
DATA_FOLDER = Path("data/raw")

# LOAD DATASETS
def load_datasets():
    parquet_files = sorted(DATA_FOLDER.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            "No Parquet datasets found inside data/raw")
    return parquet_files

# DISPLAY DATASET NAMES
def display_dataset_names(files):
    print("\nDatasets Found:\n")
    for file in files:
        print(file.name)

# DATASET SUMMARY
def dataset_summary(files):
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        print(f"\nDataset : {file.name}")
        print(f"Rows     : {df.shape[0]}")
        print(f"Columns  : {df.shape[1]}")

# COLUMN INFORMATION
def column_information(files):
    sample_df = pd.read_parquet(files[0])
    print("\n" + "=" * 80)
    print("COLUMN INFORMATION")
    print("=" * 80)
    for index, column in enumerate(sample_df.columns, start=1):
        print(f"{index:2}. {column}")

# DATA TYPES
def data_types(files):
    sample_df = pd.read_parquet(files[0])
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print(sample_df.dtypes)

# DATASET OVERVIEW
def dataset_overview(files):
    sample_df = pd.read_parquet(files[0])
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    sample_df.info()

# MEMORY USAGE
def memory_usage(files):
    sample_df = pd.read_parquet(files[0])
    memory = sample_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print("\n" + "=" * 80)
    print("MEMORY USAGE")
    print("=" * 80)
    print(f"Memory Usage : {memory:.2f} MB")

# MISSING VALUES REPORT
def check_missing_values(files):
    print("\n" + "=" * 80)
    print("MISSING VALUES REPORT")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        missing = df.isnull().sum()
        total_missing = missing.sum()
        print(f"\nDataset : {file.name}")
        print(f"Total Missing Values : {total_missing}")
        if total_missing == 0:
            print("Status : No Missing Values")
        else:
            print("\nColumns with Missing Values:")
            print(missing[missing > 0])

# DUPLICATE ROWS REPORT
def check_duplicate_rows(files):
    print("\n" + "=" * 80)
    print("DUPLICATE ROWS REPORT")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        duplicates = df.duplicated().sum()
        print(f"\nDataset : {file.name}")
        print(f"Duplicate Rows : {duplicates}")
        if duplicates == 0:
            print("Status : No Duplicate Rows")
        else:
            print("Status : Duplicate Rows Found")

# INFINITE VALUES REPORT
def check_infinite_values(files):
    print("\n" + "=" * 80)
    print("INFINITE VALUES REPORT")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        numeric_df = df.select_dtypes(include=[np.number])
        infinite_values = np.isinf(numeric_df).sum().sum()
        print(f"\nDataset : {file.name}")
        print(f"Infinite Values : {infinite_values}")
        if infinite_values == 0:
            print("Status : No Infinite Values")
        else:
            print("Status : Infinite Values Found")

# CONSTANT COLUMNS REPORT
def check_constant_columns(files):
    print("\n" + "=" * 80)
    print("CONSTANT COLUMNS REPORT")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        constant_columns = []
        for column in df.columns:
            if column == "Label":
                continue
            if df[column].nunique(dropna=False) == 1:
                constant_columns.append(column)
        print(f"\nDataset : {file.name}")
        if not constant_columns:
            print("Status : No Constant Columns")
        else:
            print("Constant Columns:")
            for column in constant_columns:
                print(f" - {column}")

# CLASS DISTRIBUTION REPORT
def class_distribution(files):
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION REPORT")
    print("=" * 80)
    for file in files:
        df = pd.read_parquet(file)
        print(f"\nDataset : {file.name}")
        print(df["Label"].value_counts())
        print("-" * 50)

def validate_dataset():
    files = load_datasets()
    display_dataset_names(files)
    dataset_summary(files)
    column_information(files)
    data_types(files)
    dataset_overview(files)
    memory_usage(files)
    check_missing_values(files)
    check_duplicate_rows(files)
    check_infinite_values(files)
    check_constant_columns(files)
    class_distribution(files)