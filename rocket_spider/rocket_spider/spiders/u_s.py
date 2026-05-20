import scrapy
import sqlite3
import hashlib
import re
import json

from readability import Document

from urllib.parse import (
    urlparse,
    urlunparse
)


class UniversalSpider2(scrapy.Spider):

    name = "universal2"

    custom_settings = {
        # Lower concurrency so you don't exhaust your network ports/DNS
        "CONCURRENT_REQUESTS": 16,  
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,

        # Give the servers (and your router) a little time to respond
        "DOWNLOAD_TIMEOUT": 30,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,

        # Add a tiny delay to stop acting like a DDoS attack
        "DOWNLOAD_DELAY": 0.5,
        "AUTOTHROTTLE_ENABLED": True,

        # Crawl depth - 0 for homepage only
        "DEPTH_LIMIT": 0,

        # Robots
        "ROBOTSTXT_OBEY": False,

        # Logging
        "LOG_LEVEL": "INFO",

        # Better user-agent
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),

        # Add this new block to spoof a real browser better!
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        },
    }

    def __init__(self):

        self.conn = sqlite3.connect(
            "search.db"
        )

        self.cursor = self.conn.cursor()

        self.pages_saved = 0

        self.setup_database()

    # =====================================================
    # START REQUESTS
    # =====================================================

    def start_requests(self):

        try:

            with open(
                "ranked_domains.json",
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                for item in data:

                    raw_domain = item.get(
                        "domain",
                        ""
                    )

                    if raw_domain.startswith("http"):
                        
                        url = raw_domain
                        
                    else:
                        
                        url = f"https://{raw_domain}"

                    yield scrapy.Request(

                        url,

                        callback=self.parse
                    )

        except Exception as e:

            self.logger.error(e)

    # =====================================================
    # DATABASE
    # =====================================================

    def setup_database(self):

        # Better SQLite performance
        self.cursor.execute(
            "PRAGMA journal_mode=WAL"
        )

        self.cursor.execute(
            "PRAGMA synchronous=NORMAL"
        )

        # Main pages table
        self.cursor.execute("""

        CREATE TABLE IF NOT EXISTS pages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            url TEXT UNIQUE,

            title TEXT,

            content TEXT,

            domain TEXT,

            content_hash TEXT UNIQUE,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            pagerank REAL DEFAULT 0,

            click_score REAL DEFAULT 0
        )

        """)

        # Full text search
        self.cursor.execute("""

        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
        USING fts5(

            title,

            content,

            url,

            domain
        )

        """)

        # Performance indexes
        self.cursor.execute("""

        CREATE INDEX IF NOT EXISTS idx_url
        ON pages(url)

        """)

        self.cursor.execute("""

        CREATE INDEX IF NOT EXISTS idx_hash
        ON pages(content_hash)

        """)

        self.cursor.execute("""

        CREATE INDEX IF NOT EXISTS idx_domain
        ON pages(domain)

        """)

        self.conn.commit()

    # =====================================================
    # TEXT CLEANING
    # =====================================================

    def clean_text(self, text):

        if not text:
            return ""

        # Remove HTML
        text = re.sub(
            r"<[^>]+>",
            " ",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =====================================================
    # HASH CONTENT
    # =====================================================

    def hash_content(self, content):

        return hashlib.md5(
            content.encode(
                errors="ignore"
            )
        ).hexdigest()

    # =====================================================
    # NORMALIZE URL
    # =====================================================

    def normalize_url(self, url):

        parsed = urlparse(url)

        cleaned = parsed._replace(

            fragment="",

            query=""
        )

        normalized = urlunparse(
            cleaned
        )

        return normalized.rstrip("/")

    # =====================================================
    # VALID URL CHECK
    # =====================================================

    def is_valid_url(self, url):

        parsed = urlparse(url)

        if parsed.scheme not in [
            "http",
            "https"
        ]:
            return False

        bad_extensions = (

            ".jpg",
            ".jpeg",
            ".png",
            ".gif",

            ".pdf",
            ".zip",

            ".svg",

            ".mp4",
            ".mp3",

            ".css",
            ".js",

            ".woff",
            ".woff2",

            ".ico",

            ".json",
            ".xml"
        )

        if parsed.path.lower().endswith(
            bad_extensions
        ):
            return False

        return True

    # =====================================================
    # SAVE PAGE
    # =====================================================

    def save_page(
        self,
        url,
        title,
        content
    ):

        url = self.normalize_url(url)

        domain = urlparse(url).netloc

        content_hash = self.hash_content(
            content
        )

        try:

            # Skip duplicate content
            self.cursor.execute("""

            SELECT 1
            FROM pages
            WHERE content_hash = ?

            """, (content_hash,))

            exists = self.cursor.fetchone()

            if exists:
                return

            # Insert page
            self.cursor.execute("""

            INSERT OR IGNORE INTO pages (

                url,

                title,

                content,

                domain,

                content_hash

            )
            VALUES (?, ?, ?, ?, ?)

            """, (

                url,

                title,

                content,

                domain,

                content_hash
            ))

            # Only continue if inserted
            if self.cursor.rowcount > 0:

                page_id = self.cursor.lastrowid

                # Insert into FTS
                self.cursor.execute("""

                INSERT INTO pages_fts (

                    rowid,

                    title,

                    content,

                    url,

                    domain

                )
                VALUES (?, ?, ?, ?, ?)

                """, (

                    page_id,

                    title,

                    content,

                    url,

                    domain
                ))

                self.pages_saved += 1

                # Commit every 100 pages
                if self.pages_saved % 100 == 0:

                    self.conn.commit()

                    self.logger.info(
                        f"Committed "
                        f"{self.pages_saved} pages"
                    )

                self.logger.info(
                    f"Saved: {url}"
                )

        except Exception as e:

            self.logger.warning(e)

    # =====================================================
    # PARSE PAGE
    # =====================================================

    def parse(self, response):

        try:

            # Only process HTML
            content_type = response.headers.get(
                "Content-Type",
                b""
            ).decode()

            if "text/html" not in content_type:
                return

            # Extract readable content
            doc = Document(
                response.text
            )

            title = self.clean_text(
                doc.title()
            )

            content = self.clean_text(
                doc.summary()
            )

            # Fallback if content is too small
            if len(content) < 50:
                
                content = title

            self.save_page(

                response.url,

                title,

                content
            )

        except Exception as e:

            self.logger.warning(e)

    # =====================================================
    # SHUTDOWN
    # =====================================================

    def closed(self, reason):

        self.conn.commit()

        self.conn.close()

        print(
            "Database closed safely"
        )