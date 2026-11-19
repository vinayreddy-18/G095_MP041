
import pandas as pd
from pathlib import Path

# Read original dataset
df = pd.read_csv("data/train.csv")

print("Original rows:", len(df))

# 1. Remove duplicate rows
df = df.drop_duplicates().copy()

# 2. Correct distance units in Market
market = df["zone"] == "Market"
df.loc[market, "distance_km"] = (
    df.loc[market, "distance_km"] / 1000
)

# 3. Correct target values typed 10 times too large
typos = df["fare_rs"] > 1000
df.loc[typos, "fare_rs"] = (
    df.loc[typos, "fare_rs"] / 10
)

# 4. Handle stuck fuel-price readings
stuck = df["date"].between("2026-03-26", "2026-04-09")

# Calculate median from readings outside the stuck period
median_fuel = df.loc[~stuck, "fuel_price_rs"].median()
df.loc[stuck, "fuel_price_rs"] = median_fuel

# 5. Remove the leaking receipt column
df = df.drop(columns=["fare_paid_receipt_rs"])

# 6. Save cleaned data separately
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/train_clean.csv", index=False)

print("Cleaned rows:", len(df))
print("Duplicate rows remaining:", df.duplicated().sum())
print("Maximum distance:", df["distance_km"].max())
print("Maximum fare:", df["fare_rs"].max())
print("Cleaned dataset saved successfully!")
