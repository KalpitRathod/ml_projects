from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import sqlite_vec
from sentence_transformers import SentenceTransformer
import sqlite3
import re
import time


# =========================================================
# CONSTANTS
# =========================================================

MAX_QUERY_LENGTH = 200
MAX_RESULTS = 20
CLICK_COOLDOWN = 3600  # 1 hour per (IP, URL) pair

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# RATE LIMIT STORE (in-memory, per-process)
# =========================================================

# Maps (client_ip, url) -> timestamp of last allowed click
_click_store: dict[tuple[str, str], float] = {}


def is_click_allowed(ip: str, url: str) -> bool:
    """
    Returns True (and records the click) if this IP hasn't
    clicked this URL within CLICK_COOLDOWN seconds.
    Prunes stale entries when the store grows too large.
    """
    key = (ip, url)
    now = time.time()

    if now - _click_store.get(key, 0) < CLICK_COOLDOWN:
        return False

    _click_store[key] = now

    # Prune stale entries to prevent unbounded memory growth
    if len(_click_store) > 10_000:
        cutoff = now - CLICK_COOLDOWN
        stale = [k for k, v in _click_store.items() if v < cutoff]
        for k in stale:
            del _click_store[k]

    return True


# =========================================================
# DATABASE — per-request connection (thread-safe)
# =========================================================

def _make_conn() -> sqlite3.Connection:

    conn = sqlite3.connect(
        "search.db",
        check_same_thread=False
    )

    conn.enable_load_extension(True)

    sqlite_vec.load(conn)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("PRAGMA synchronous=NORMAL")

    return conn


def get_db():
    """
    FastAPI dependency: opens a fresh SQLite connection for
    each request and closes it when the request finishes.
    Avoids the shared-connection concurrency bug.
    """
    conn = _make_conn()
    try:
        yield conn
    finally:
        conn.close()


# =========================================================
# LIFESPAN — replaces deprecated @app.on_event
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────
    conn = _make_conn()
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_url
        ON pages(url)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pagerank
        ON pages(pagerank)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_created
        ON pages(created_at)
    """)
    conn.commit()
    conn.close()

    yield

    # ── Shutdown ─────────────────────────────────────────
    print("Server shutdown — all per-request connections already closed.")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HOME PAGE — served from templates/index.html
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# =========================================================
# SEARCH API
# =========================================================

@app.get("/search")
def search(
    q: str,
    page: int = 0,
    db: sqlite3.Connection = Depends(get_db),
):
    start_time = time.time()

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if len(q) > MAX_QUERY_LENGTH:
        return []

    q = re.sub(r"[^\w\s]", " ", q).strip()

    if not q:
        return []

    # =====================================================
    # PAGINATION
    # =====================================================

    offset = page * MAX_RESULTS

    # =====================================================
    # GENERATE QUERY EMBEDDING
    # =====================================================

    query_embedding = model.encode(q)

    # =====================================================
    # QUERY
    # =====================================================

    cursor = db.cursor()

    cursor.execute("""

    WITH semantic AS (

        SELECT

            rowid,

            distance

        FROM page_vectors

        WHERE embedding MATCH ?

        ORDER BY distance

        LIMIT 200
    )

    SELECT

        p.title,

        p.url,

        snippet(
            pages_fts,
            1,
            '<b style="color:#facc15">',
            '</b>',
            '...',
            24
        ) AS snippet,

        bm25(pages_fts) AS bm25_score,

        semantic.distance AS semantic_distance,

        p.pagerank,

        p.click_score,

        p.created_at,

        (

            ------------------------------------------------
            -- SEMANTIC SIMILARITY
            ------------------------------------------------

            (
                1.0 / (semantic.distance + 1.0)
            ) * 0.45

            +

            ------------------------------------------------
            -- BM25 KEYWORD RELEVANCE
            ------------------------------------------------

            (
                -bm25(pages_fts)
            ) * 0.30

            +

            ------------------------------------------------
            -- PAGERANK
            ------------------------------------------------

            (
                p.pagerank
            ) * 0.15

            +

            ------------------------------------------------
            -- CLICK SCORE
            ------------------------------------------------

            (
                MIN(p.click_score, 100)
            ) * 0.05

            +

            ------------------------------------------------
            -- FRESHNESS
            ------------------------------------------------

            (

                1.0 / (

                    LOG(

                        julianday('now')
                        - julianday(p.created_at)
                        + 2

                    )

                )

            ) * 0.05

        ) AS final_score

    FROM semantic

    JOIN pages p
    ON p.id = semantic.rowid

    JOIN pages_fts
    ON pages_fts.rowid = p.id

    WHERE pages_fts MATCH ?

    ORDER BY final_score DESC

    LIMIT ? OFFSET ?

    """, (

        sqlite_vec.serialize_float32(query_embedding),

        q,

        MAX_RESULTS,

        offset

    ))

    rows = cursor.fetchall()

    # =====================================================
    # FORMAT RESULTS
    # =====================================================

    results = [

        {
            "title": row["title"],
            "url": row["url"],
            "snippet": row["snippet"],
        }

        for row in rows
    ]

    # =====================================================
    # LOGGING
    # =====================================================

    print(

        f"Search '{q}' page={page} → "
        f"{len(results)} results in "
        f"{time.time() - start_time:.4f}s"

    )

    return results

# =========================================================
# CLICK TRACKING
# =========================================================

@app.post("/click")
def click(
    url: str,
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
):
    client_ip = request.client.host if request.client else "unknown"

    # ── Rate limit: 1 click per IP per URL per hour ───────
    if not is_click_allowed(client_ip, url):
        return {"status": "rate_limited"}

    cursor = db.cursor()

    # ── Guard: only update URLs that actually exist ───────
    cursor.execute(
        "SELECT id FROM pages WHERE url = ? LIMIT 1",
        (url,)
    )
    if cursor.fetchone() is None:
        return {"status": "not_found"}

    cursor.execute("""
        UPDATE pages
        SET click_score = click_score + 1
        WHERE url = ?
    """, (url,))

    db.commit()

    return {"status": "ok"}