import pandas as pd

df = pd.read_csv("data/raw/loan_Default.csv")
print(df)

print(df.head())
print(df.shape)
print(df.info())
print(df.columns)