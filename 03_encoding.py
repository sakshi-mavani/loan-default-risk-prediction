import pandas as pd 

# load cleaned data 

df = pd.read_csv("data/processed/loan_default_cleaned.csv")
print(df)

# check target column

print(df['Status'].dtype)
print(df['Status'].value_counts)

# status already numeric (0/1), no re-encoding needed

df['Status'] = df['Status'].astype(int)

# select categorical columns 

cat_cols = df.select_dtypes(include= ['object']).columns
print(cat_cols)

# one-hot encode categorical columns 

df_encoded = pd.get_dummies(
    df,
    columns= cat_cols,
    drop_first= True
)

print(df_encoded.shape)

# save encoded data 

df_encoded.to_csv("data/processed/loan_default_encoded.csv", index = False)
print("Encoded file saved")

