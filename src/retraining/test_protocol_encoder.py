import joblib

encoder = joblib.load("models/live_protocol_encoder.pkl")

print("Classes:")
print(encoder.classes_)

print("TCP ->", encoder.transform(["TCP"]))
print("UDP ->", encoder.transform(["UDP"]))