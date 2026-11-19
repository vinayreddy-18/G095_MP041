import pandas as pd

df = pd.read_csv("data/train.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDataset Summary:")
print(df.describe(include="all"))


# Check 1: Data leakage
print("\n1. DATA LEAKAGE")
holdout = pd.read_csv("data/holdout_features.csv")
print("Columns missing from holdout:")
print(set(df.columns) - set(holdout.columns))

# Check 2: Unit mix-up
print("\n2. UNIT MIX-UP")
print(df.groupby("zone")["distance_km"].agg(["min", "median", "max"]))

# Check 3: Sentinel values
print("\n3. SENTINEL VALUES")
for column in df.select_dtypes(include="number").columns:
    print(column, df[column].value_counts().head(5).to_dict())

# Check 4: Duplicate rows
print("\n4. DUPLICATE ROWS")
print(df.duplicated().sum())

# Check 5: Stuck sensor
print("\n5. STUCK SENSOR")
sensor = df[
    (df["date"] >= "2026-03-26") &
    (df["date"] <= "2026-04-09")
]
print(sensor.groupby("date")["fuel_price_rs"].agg(["min", "max", "count"]))

# Check 6: Target typos
print("\n6. TARGET TYPOS")
print(df["fare_rs"].sort_values(ascending=False).head(15))
print("Fares above Rs 500:", (df["fare_rs"] > 500).sum())

# Check 7: Spelling variants
print("\n7. SPELLING VARIANTS")
print(df["booking"].value_counts())
print(df["zone"].value_counts())
