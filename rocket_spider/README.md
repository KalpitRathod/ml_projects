# Rocket Spider

A Scrapy-based intelligent web crawler and search system with embedding-powered search.

This project crawls websites, stores indexed content in SQLite, generates embeddings, and provides a searchable interface through Flask.

Repository:
https://github.com/KalpitRathod/ml_projects

---

## Features

- Universal web crawler using Scrapy
- SQLite-based storage
- Embedding generation for semantic search
- Flask frontend for searching indexed pages
- Ranked domain support
- Configurable spiders

---

## Project Structure

```bash
rocket_spider/
│
├── app.py                     # Flask application
├── embedder.py                # Embedding generation logic
├── search.py                  # Search engine logic
├── ranked_domains.json        # Some good domain to crawl see the spider universal crawler
├── scrapy.cfg                 # Scrapy config
├── search.db                  # SQLite database (You have to sun spider first to create it. It will get big I have crawled 19000+ pages (1.5 GB db file) in one day.)
│
├── templates/
│   └── index.html             # Frontend UI
│
└── rocket_spider/
    ├── settings.py
    ├── items.py
    ├── pipelines.py
    ├── middlewares.py
    │
    └── spiders/
        ├── universal_spider.py # This file have two codes
        └── u_s.py # this have one code
````

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/KalpitRathod/ml_projects.git
cd ml_projects
```

---

### 2. Create Virtual Environment

```bash
python3 -m venv scrapyenv
source scrapyenv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Spider

Example:
this file has 2 code separeted by (---) so uncomment one section to run it.
```bash
cd rocket_spider
scrapy crawl universal
```

Or:

```bash
scrapy crawl universal2
```

---

## Run embedder.py

```bash
python embedder.py
```

---

## Running the Flask Search App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Database

The project uses SQLite:

```text
search.db
```

---

## Example Workflow

1. Run spider
2. Store crawled content into SQLite
3. Generate embeddings
4. Start Flask app
5. Search indexed content

---

## License

MIT License