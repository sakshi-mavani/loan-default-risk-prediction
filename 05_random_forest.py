import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (accuracy_score, classification_report, roc_auc_score, confusion_matrix)

print("Script started...")

df = pd.read_csv("data/processed/loan_default_encoded.csv")
print("Data loaded:", df.shape)

X = df.drop("Status", axis= 1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size= 0.2, 
                                                    random_state= 42, 
                                                    stratify= y
)


print("Train-test-split done")

model = RandomForestClassifier(n_estimators= 200,
                               max_depth= 10,
                               random_state= 42,
                               n_jobs= -1
)

model.fit(X_train, y_train)
print("Random Forest Trained")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", accuracy)
print("\nROC-AUC:", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import os 
os.makedirs("model", exist_ok = True)

joblib.dump(model, "model/loan_default_rf.pkl")
print("\nModel saved successfully")

print("Script ended")