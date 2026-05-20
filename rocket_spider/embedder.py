import sqlite3
import sqlite_vec

from pathlib import Path

from sentence_transformers import SentenceTransformer


# =========================================================
# MODEL
# =========================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"    
)


# =========================================================
# DATABASE
# =========================================================

DB_PATH = Path(__file__).resolve().parent / "search.db"

db = sqlite3.connect(DB_PATH)

db.enable_load_extension(True)

sqlite_vec.load(db)

cursor = db.cursor()


# =========================================================
# VECTOR TABLE
# =========================================================

cursor.execute("""

CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors
USING vec0(
    embedding float[384]
)

""")


# =========================================================
# ONLY LOAD PAGES WITHOUT EMBEDDINGS
# =========================================================

rows = cursor.execute("""

SELECT

    p.id,
    p.title,
    p.content

FROM pages p

LEFT JOIN page_vectors v
ON p.id = v.rowid

WHERE v.rowid IS NULL

""").fetchall()


print(f"Found {len(rows)} pages needing embeddings")


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

count = 0

for row in rows:

    page_id = row[0]

    title = row[1] or ""

    content = row[2] or ""

    text = f"{title}\n\n{content[:5000]}"

    embedding = model.encode(text)

    cursor.execute("""

    INSERT INTO page_vectors(
        rowid,
        embedding
    )
    VALUES (?, ?)

    """, (
        page_id,
        sqlite_vec.serialize_float32(embedding)
    ))

    count += 1

    if count % 100 == 0:

        db.commit()

        print(f"Embedded {count} pages")


db.commit()

print("DONE")