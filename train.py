import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "predictive_maintenance_integer.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical target variable
label_encoder = LabelEncoder()
df["Failure Mode"] = label_encoder.fit_transform(df["Failure Mode"])

# Split features and target
X = df.drop(columns=["Failure Mode"])
y = df["Failure Mode"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance (compute class weights)
class_weights = {i: len(y) / (len(np.unique(y)) * np.bincount(y)[i]) for i in np.unique(y)}
scale_pos_weight = [class_weights[i] for i in y_train]

# Train XGBoost model
model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y)), eval_metric="mlogloss",
                          scale_pos_weight=scale_pos_weight)

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "xgboost_failure_mode.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model saved successfully!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))
