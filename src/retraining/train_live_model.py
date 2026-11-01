import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_parquet("data/processed/live_features_dataset.parquet")

# ==========================================
# Convert Protocol to Numeric
# ==========================================

protocol_map = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1
}

if df["Protocol"].dtype == object:
    df["Protocol"] = df["Protocol"].map(protocol_map)

df["Protocol"] = df["Protocol"].fillna(0).astype(int)

# ==========================================
# Features and Label
# ==========================================

X = df.drop("Label", axis=1)
y = df["Label"]

# ==========================================
# Encode Labels
# ==========================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Scale Features
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Train Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# Evaluation
# ==========================================

predictions = model.predict(X_test)

print("=" * 60)
print("Accuracy :", accuracy_score(y_test, predictions))
print("=" * 60)

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "models/live_random_forest_model.pkl")
joblib.dump(scaler, "models/live_scaler.pkl")
joblib.dump(label_encoder, "models/live_label_encoder.pkl")

print("\nLive AI Model Saved Successfully!")