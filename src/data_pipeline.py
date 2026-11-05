
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


def clean_customers():
    customers = pd.read_csv(RAW_DIR / "customers.csv")

    # Standardize city names
    customers["city"] = (
        customers["city"].astype("string").str.strip().str.title()
    )

    # Standardize gender values
    gender = customers["gender"].astype("string").str.strip().str.lower()

    customers["gender"] = gender.map({
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }).fillna("unknown")

    # Convert age to numeric and invalidate ages outside 10-100
    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )
    customers.loc[
        ~customers["age"].between(10, 100), "age"
    ] = float("nan")

    # Fill missing age with the median age of the same city
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)

    # Convert signup_date to datetime
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    return customers


def clean_products():
    products = pd.read_csv(RAW_DIR / "products.csv")

    for column in ["product_name", "category"]:
        products[column] = products[column].astype("string").str.strip()

    return products


def clean_orders(products):
    orders = pd.read_csv(RAW_DIR / "orders.csv")

    # Remove exact duplicate rows only
    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()

    # Convert timestamps and sort records chronologically
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    if orders["order_ts"].isna().any():
        raise ValueError("Some order timestamps could not be parsed.")

    # Convert amount and delivery columns to numeric
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Negative delivery times are invalid
    orders.loc[
        orders["delivery_minutes"] < 0, "delivery_minutes"
    ] = float("nan")

    # Attach product price and category
    orders = orders.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    if orders["price_inr"].isna().any():
        raise ValueError("An order has no matching product price.")

    # Keep only the latest record for each order_id
    orders = orders.sort_values(
        "order_ts", kind="mergesort"
    )
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    ).copy()

    # Repair missing or invalid order amounts
    missing_amount = orders["order_amount"].isna()

    calculated_amount = (
        orders["quantity"]
        * orders["price_inr"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    if calculated_amount[missing_amount].isna().any():
        raise ValueError("Could not calculate some order amounts.")

    if (calculated_amount[missing_amount] <= 0).any():
        raise ValueError("Calculated order amount is not positive.")

    orders.loc[missing_amount, "order_amount"] = (
        calculated_amount[missing_amount]
    )

    repaired_amounts = int(missing_amount.sum())

    # Fill missing delivery time for completed orders
    # using the median for the corresponding product category
    completed = orders["status"].eq("completed")

    category_medians = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .median()
    )

    missing_completed = (
        completed & orders["delivery_minutes"].isna()
    )

    orders.loc[missing_completed, "delivery_minutes"] = (
        orders.loc[missing_completed, "category"]
        .map(category_medians)
    )

    if orders.loc[missing_completed, "delivery_minutes"].isna().any():
        raise ValueError(
            "Could not find a delivery-time median for a category."
        )

    # Remove helper columns
    orders = orders.drop(columns=["price_inr", "category"])

    print("\nOrder cleaning details:")
    print("Exact duplicate rows removed:", exact_duplicates)
    print("Order amounts repaired:", repaired_amounts)
    print("Final order rows:", len(orders))
    print("\nFinal status counts:")
    print(orders["status"].value_counts().to_string())
    print(
        "\nMissing delivery times:",
        orders["delivery_minutes"].isna().sum()
    )
    print(
        "Completed revenue:",
        round(orders.loc[completed, "order_amount"].sum(), 2)
    )

    return orders




def save_to_database(customers, products, orders):
    db_path = Path("data/instamart.db")

    with sqlite3.connect(db_path) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

    print("\nSQLite database created successfully!")
    print("Database location:", db_path)
import sqlite3


def save_to_database(customers, products, orders):
    db_path = Path("data/instamart.db")

    with sqlite3.connect(db_path) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

    print("\nSQLite database created successfully!")
    print("Database location:", db_path)


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    customers = clean_customers()
    products = clean_products()
    orders = clean_orders(products)

    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    print("\nCleaning completed successfully.")
    print("Customers:", len(customers))
    print("Products:", len(products))
    print("Orders:", len(orders))
    print("Saved location:", CLEAN_DIR)

    save_to_database(customers, products, orders)


if __name__ == "__main__":
    main()