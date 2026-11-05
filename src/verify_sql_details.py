import sqlite3
import pandas as pd

connection = sqlite3.connect("data/instamart.db")

# Q6: Top spending customer in each city
q6 = """
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS rank_in_city,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS dense_rank_in_city,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC, customer_id
    ) AS row_num_in_city
FROM customer_spend
"""

ranking = pd.read_sql_query(q6, connection)

print("\n========== Q6: TOP CUSTOMER IN EACH CITY ==========")
print(
    ranking[ranking["rank_in_city"] == 1]
    .sort_values("city")
    .to_string(index=False)
)

tie_count = (
    ranking["rank_in_city"] != ranking["row_num_in_city"]
).sum()

print("\nRows where RANK differs from ROW_NUMBER:", tie_count)


# Q7: Final daily cumulative revenue
q7 = """
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    SUM(revenue) OVER (
        ORDER BY day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day
"""

daily = pd.read_sql_query(q7, connection)

print("\n========== Q7: DAILY REVENUE ==========")
print("Number of daily rows:", len(daily))
print("Final cumulative revenue:", round(daily.iloc[-1]["cumulative_revenue"], 2))

print("\n========== COMPLETED CUSTOMERS BY CITY ==========")

city_check = pd.read_sql_query("""
SELECT
    c.city,
    COUNT(DISTINCT o.customer_id) AS customers_with_completed_orders
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY c.city
""", connection)

print(city_check.to_string(index=False))

connection.close()