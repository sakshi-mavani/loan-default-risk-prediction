import pandas as pd 

df = pd.read_csv("data/raw/Loan_Default.csv")
print(df)

# check missinng value 

missing_percent = (df.isnull().sum()) / len(df) * 100
print(missing_percent.sort_values(ascending= False))

# separete numerical and categorical columns

num_cols = df.select_dtypes(include= ['int64', 'float64']).columns
print("num_cols:", num_cols)

cat_cols = df.select_dtypes(include= ['object']).columns
print("cat_cols:", cat_cols)

# Fill missing values in numerical columns using median 

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("\nNumerical Columns missing values filled using median")

# Fill missing values in categorical columns using mode

df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
print("\nNumerical Columns missing values filled using mode")

# Final Verification

print("\nRemaining missing values (should be 0):")
print(df.isnull().sum().max())

# drop columns

drop_cols = ['ID'] if 'ID' in df.columns else []
df = df.drop(columns= drop_cols)
print("Dropped columns:", drop_cols)

df.to_csv("data/processed/loan_default_cleaned.csv", index = False)
print("File saved successfully..")