import pandas as pd
import joblib

# load data 

df = pd.read_csv("data/processed/loan_default_encoded.csv")

X = df.drop("Status", axis= 1)
y = df["Status"]
 
# load trained model 

model = joblib.load("model/loan_default_rf.pkl")

# feature importance 

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by = "importance", ascending= False)

print(feature_importance_df.head(15))

feature_importance_df.to_csv("data/processed/feature_importance.csv", index= False)
print("Feature importance saved")

