from model_utils import load_processed_data, save_model
from logistic_regression import train_logistic_regression
from evaluate_model import evaluate_model


def train():

    X_train, X_test, y_train, y_test = load_processed_data()

    model = train_logistic_regression(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, "logistic_regression_model.pkl")


if __name__ == "__main__":
    train()