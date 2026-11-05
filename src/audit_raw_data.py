import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

files = ["customers.csv", "products.csv", "orders.csv"]

for filename in files:
    print("\n" + "=" * 65)
    print("FILE:", filename)

    df = pd.read_csv(RAW_DIR / filename)

    print("Shape:", df.shape)

    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })

    print(audit.to_string())


# Customer-specific audit
customers = pd.read_csv(RAW_DIR / "customers.csv")

print("\n========== CUSTOMER ANOMALIES ==========")
print("Blank ages:", customers["age"].isna().sum())

age = pd.to_numeric(customers["age"], errors="coerce")
print("Ages outside 10-100:", ((age < 10) | (age > 100)).sum())

print("Raw city spellings:", customers["city"].nunique())
print(
    "Raw nonblank gender spellings:",
    customers["gender"].dropna().nunique()
)
print("Blank genders:", customers["gender"].isna().sum())


# Order-specific audit
orders = pd.read_csv(RAW_DIR / "orders.csv")

print("\n========== ORDER ANOMALIES ==========")
print("Exact duplicate rows:", orders.duplicated().sum())
print("Distinct order IDs:", orders["order_id"].nunique())
print("Pending rows:", (orders["status"] == "pending").sum())

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
print("Nonnumeric order amounts:", amount.isna().sum())

delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
print("Negative delivery times:", (delivery < 0).sum())
print("Blank delivery times:", delivery.isna().sum())

print("\nRaw data audit completed successfully.")