from sklearn.linear_model import LogisticRegression
import time


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """

    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Number of Features : {X_train.shape[1]}")

    # Create model
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    # Start timer
    start_time = time.time()

    # Train model
    model.fit(X_train, y_train)

    # End timer
    end_time = time.time()

    training_time = end_time - start_time

    print("\nLogistic Regression training completed successfully.")
    print(f"Training Time : {training_time:.2f} seconds")

    return model