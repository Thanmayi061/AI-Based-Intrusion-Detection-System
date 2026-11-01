from model_utils import load_processed_data, save_model
from decision_tree import train_decision_tree
from evaluate_model import evaluate_model


def train():

    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train Decision Tree
    model = train_decision_tree(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, "decision_tree_model.pkl")


if __name__ == "__main__":
    train()