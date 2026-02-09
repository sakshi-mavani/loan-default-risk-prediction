import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier


print("Script started")

# 1. Load encoded data
df = pd.read_csv("data/processed/loan_default_encoded.csv")
print("Data loaded:", df.shape)

# 2. Separate features and target
X = df.drop("Status", axis=1)
y = df["Status"]

print("Target distribution:")
print(y.value_counts())

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split done")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 4. 🔥 FIX: clean column names for XGBoost
X_train.columns = X_train.columns.str.replace(r"[\\[\\]<]", "_", regex=True)
X_test.columns = X_test.columns.str.replace(r"[\\[\\]<]", "_", regex=True)

# 5. XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

print("XGBoost model created")

# FIX column names for XGBoost
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train.columns = X_train.columns.str.replace(r'[\[\]<]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[\[\]<]', '_', regex=True)


# 6. Train model
model.fit(X_train, y_train)
print("Model trained")

# 7. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluation
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", accuracy)
print("ROC-AUC:", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save model (deployment ready)
joblib.dump(model, "model/loan_default_xgboost.pkl")
print("\nModel saved successfully")


joblib.dump(model, "model/final_model.pkl")
joblib.dump(X_train.columns.tolist(), "model/model_features.pkl")


print("Script ended")
