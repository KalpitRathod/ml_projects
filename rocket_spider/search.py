import sqlite3


conn = sqlite3.connect("search.db")

cursor = conn.cursor()

query = input("Search: ")

cursor.execute("""
SELECT title, url
FROM pages_fts
WHERE pages_fts MATCH ?
LIMIT 10
""", (query,))

results = cursor.fetchall()

for row in results:

    print("\n")
    print(row[0])
    print(row[1])