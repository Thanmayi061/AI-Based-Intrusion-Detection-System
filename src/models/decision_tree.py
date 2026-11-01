from sklearn.tree import DecisionTreeClassifier
import time


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.
    """

    print("\n" + "=" * 60)
    print("TRAINING DECISION TREE MODEL")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Number of Features : {X_train.shape[1]}")

    model = DecisionTreeClassifier(
        random_state=42
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    print("\nDecision Tree training completed successfully.")
    print(f"Training Time : {training_time:.2f} seconds")

    return model