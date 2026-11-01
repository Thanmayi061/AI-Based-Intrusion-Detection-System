from sklearn.ensemble import RandomForestClassifier
import time


def train_random_forest(X_train, y_train):
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Number of Features : {X_train.shape[1]}")

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Start timer
    start_time = time.time()

    # Train model
    model.fit(X_train, y_train)

    # End timer
    end_time = time.time()

    training_time = end_time - start_time

    print("\nRandom Forest training completed successfully.")
    print(f"Training Time : {training_time:.2f} seconds")

    return model