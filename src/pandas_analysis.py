import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/instamart.db")

connection = sqlite3.connect(DB_PATH)

# Load clean tables
customers = pd.read_sql_query("SELECT * FROM customers", connection)
products = pd.read_sql_query("SELECT * FROM products", connection)
orders = pd.read_sql_query("SELECT * FROM orders", connection)

connection.close()

# Keep completed orders only
completed = orders[orders["status"] == "completed"].copy()

# Convert order amount to numeric
completed["order_amount"] = pd.to_numeric(completed["order_amount"])

print("\n========== 1. SQL Q2 CROSS-CHECK ==========")

merged = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).merge(
    products[["product_id", "category"]],
    on="product_id"
)

pivot = pd.pivot_table(
    merged,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2))

sql_total = completed["order_amount"].sum()
pandas_total = pivot.loc["Grand Total", "Grand Total"]

print("\nSQL total revenue:", round(sql_total, 2))
print("Pandas total revenue:", round(pandas_total, 2))
assert round(sql_total, 2) == round(pandas_total, 2)


print("\n========== 2. CITY-WISE ANALYSIS ==========")

city_analysis = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_amount", "count")
)

print(city_analysis.round(2))


print("\n========== 3. ORDER AMOUNT STATISTICS ==========")

amount = completed["order_amount"]

print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Skewness:", round(amount.skew(), 2))
print("Q1:", round(amount.quantile(0.25), 2))
print("Q3:", round(amount.quantile(0.75), 2))


print("\n========== 4. OUTLIER ANALYSIS ==========")

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below = (amount < lower_fence).sum()
above = (amount > upper_fence).sum()

print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", below)
print("Outliers above upper fence:", above)


print("\n========== 5. CORRELATION MATRIX ==========")

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = completed[correlation_columns].corr()

print(correlation.round(3))

print("\nPandas analysis completed successfully.")