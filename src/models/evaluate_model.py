from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD LABEL ENCODER
# ==========================================

encoder = joblib.load(Path("models") / "label_encoder.pkl")

# ==========================================
# CREATE REPORTS FOLDER
# ==========================================

REPORTS_FOLDER = Path("reports")
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"Accuracy : {accuracy:.4f}")

    # Classification Report
    report = classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )

    print("\nClassification Report\n")
    print(report)

    # Save Classification Report
    with open(REPORTS_FOLDER / "classification_report.txt", "w") as file:
        file.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix\n")
    print(cm)

    # Normalize Confusion Matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]

    # Plot Heatmap
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(REPORTS_FOLDER / "confusion_matrix.png")

    plt.show()

    print("\nConfusion Matrix saved successfully.")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def plot_feature_importance(model, X_train):

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save complete feature importance
    feature_df.to_csv(
        REPORTS_FOLDER / "feature_importance.csv",
        index=False
    )

    # Top 20 features
    top_features = feature_df.head(20)

    print("\n" + "=" * 80)
    print("TOP 20 IMPORTANT FEATURES")
    print("=" * 80)
    print(top_features)

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(REPORTS_FOLDER / "feature_importance.png")

    plt.show()

    print("\nFeature Importance saved successfully.")