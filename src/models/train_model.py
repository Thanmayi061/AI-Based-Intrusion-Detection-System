from model_utils import load_processed_data, save_model
from random_forest import train_random_forest
from evaluate_model import evaluate_model, plot_feature_importance


def train():

    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train Random Forest
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(model, X_train)

    # Save the trained model
    save_model(model, "random_forest_model.pkl")


if __name__ == "__main__":
    train()