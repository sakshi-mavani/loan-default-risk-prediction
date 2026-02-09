import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Script started")

df = pd.read_csv("data/processed/loan_default_encoded.csv")
print("Data loaded:", df.shape)

X = df.drop("Status", axis=1)
y = df["Status"]

print("Target distribution:")
print(y.value_counts())

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

model = LogisticRegression(max_iter=1000, solver= "liblinear", class_weight= "balanced")
print("Model created")

model.fit(X_train, y_train)
print("Model trained")

y_pred = model.predict(X_test)
print("Prediction done")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Script ended")
