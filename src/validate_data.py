import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/instamart.db")

connection = sqlite3.connect(DB_PATH)

# Read tables
customers = pd.read_sql_query("SELECT * FROM customers", connection)
products = pd.read_sql_query("SELECT * FROM products", connection)
orders = pd.read_sql_query("SELECT * FROM orders", connection)

# Checkpoints
assert len(customers) == 1200, "Customer row count mismatch"
assert len(products) == 40, "Product row count mismatch"
assert len(orders) == 12309, "Order row count mismatch"

assert (orders["status"] == "completed").sum() == 11110, "Completed count mismatch"
assert (orders["status"] == "cancelled").sum() == 1199, "Cancelled count mismatch"
assert (orders["status"] == "pending").sum() == 0, "Pending orders found"

assert orders["order_amount"].isna().sum() == 0, "Missing order amounts found"

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

assert completed["delivery_minutes"].isna().sum() == 0, "Completed delivery times missing"
assert cancelled["delivery_minutes"].isna().sum() == 1199, "Cancelled delivery check failed"

assert round(completed["order_amount"].sum(), 2) == 2998668.80, "Revenue mismatch"

assert customers["age"].isna().sum() == 0, "Missing customer ages found"
assert customers["city"].nunique() == 6, "City count mismatch"
assert set(customers["gender"].unique()) == {"female", "male", "unknown"}, "Gender values mismatch"

connection.close()

print("ALL CHECKPOINTS PASSED")