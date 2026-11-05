import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

connection = sqlite3.connect(DB_PATH)

sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

print(f"Total SQL queries found: {len(queries)}")

for number, query in enumerate(queries, start=1):
    result = pd.read_sql_query(query, connection)

    print(f"\n{'=' * 50}")
    print(f"Q{number}")
    print(f"Rows returned: {len(result)}")
    print(result.head(5).to_string(index=False))

connection.close()

print("\nAll SQL queries executed successfully.")