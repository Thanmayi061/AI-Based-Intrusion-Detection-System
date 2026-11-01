from pathlib import Path
import joblib
import pandas as pd

# Load trained objects
MODEL_PATH = Path("models") / "random_forest_model.pkl"
SCALER_PATH = Path("models") / "scaler.pkl"
ENCODER_PATH = Path("models") / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)


def predict_attack(sample_df):
    """
    Predict attack type from one or more samples.
    """

    # Scale features
    scaled_array = scaler.transform(sample_df)

    # Convert back to DataFrame to preserve feature names
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=sample_df.columns,
        index=sample_df.index
    )

    # Predict
    predictions = model.predict(scaled_df)

    # Convert numeric labels to attack names
    attack_names = encoder.inverse_transform(predictions)

    return attack_names