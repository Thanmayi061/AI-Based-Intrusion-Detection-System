import pandas as pd

def check_features():

    # Load processed training dataset
    X_train = pd.read_parquet("data/processed/X_train.parquet")

    print("=" * 80)
    print("FEATURES USED FOR TRAINING")
    print("=" * 80)

    print(f"\nTotal Features: {len(X_train.columns)}\n")

    for i, column in enumerate(X_train.columns, start=1):
        print(f"{i}. {column}")

if __name__ == "__main__":
    check_features()