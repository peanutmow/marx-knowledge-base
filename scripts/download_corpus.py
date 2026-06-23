"""
scripts/download_corpus.py
──────────────────────────
Scrapes works listed in corpus_config.json from the Marxists Internet Archive
and saves them as plain-text files with JSON sidecar metadata.

Usage:
    python scripts/download_corpus.py               # download everything
    python scripts/download_corpus.py --author Marx  # one author only
    python scripts/download_corpus.py --id marx-capital-v1  # one work only
    python scripts/download_corpus.py --force        # re-download already-saved works
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── allow running as `python scripts/download_corpus.py` from project root ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

# ── Tags whose text content we want to keep ─────────────────────────────────
KEEP_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"}

# ── Tags / classes to strip entirely (navigation, boilerplate) ───────────────
STRIP_TAGS = {
    "script", "style", "nav", "header", "footer", "form",
    "button", "input", "select", "noscript",
}


def slugify(text: str) -> str:
    """Convert text to a safe filename component."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; MarxKnowledgeBaseBot/1.0; "
            "personal research; +https://github.com/user/marx-knowledge-base)"
        )
    })
    return s


def fetch_html(session: requests.Session, url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        r = session.get(url, timeout=cfg.REQUEST_TIMEOUT_S)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, "lxml")
    except Exception as exc:
        print(f"  ⚠  Failed to fetch {url}: {exc}")
        return None


def extract_text(soup: BeautifulSoup) -> str:
    """
    Pull readable text from an MIA page.
    Strips navigation, scripts, and inline link clutter; preserves headings + paragraphs.
    """
    # Remove unwanted tags in-place
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    lines = []
    for el in soup.find_all(True):
        if el.name not in KEEP_TAGS:
            continue
        text = el.get_text(separator=" ", strip=True)
        # Skip very short fragments (footnote numbers, stray symbols)
        if len(text) < 15:
            continue
        # Collapse internal whitespace
        text = re.sub(r"\s+", " ", text)
        lines.append(text)

    return "\n\n".join(lines)


def find_chapter_links(soup: BeautifulSoup, index_url: str) -> list[str]:
    """
    Given an index/table-of-contents page, return the list of chapter page URLs
    that live in the same directory on MIA.
    """
    base = index_url.rsplit("/", 1)[0] + "/"
    base_parsed = urlparse(base)

    seen = set()
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Resolve relative URLs
        full = urljoin(index_url, href)
        parsed = urlparse(full)

        # Must be on marxists.org and in the same directory (or a sub-path)
        if parsed.netloc != "www.marxists.org":
            continue
        if not parsed.path.startswith(base_parsed.path):
            continue
        # Only .htm / .html pages (not images, PDFs, etc.)
        if not re.search(r"\.html?$", parsed.path, re.IGNORECASE):
            continue
        # Exclude the index page itself
        if full == index_url or full.rstrip("/") == index_url.rstrip("/"):
            continue
        # Avoid duplicates and fragment-only variants
        clean = parsed._replace(fragment="").geturl()
        if clean not in seen:
            seen.add(clean)
            links.append(clean)

    return links


def download_work(
    session: requests.Session,
    work: dict,
    force: bool = False,
) -> bool:
    """
    Download one work. Returns True on success.
    """
    author_dir = Path(cfg.DATA_DIR) / slugify(work["author"])
    author_dir.mkdir(parents=True, exist_ok=True)

    text_path = author_dir / f"{work['id']}.txt"
    meta_path = author_dir / f"{work['id']}.json"

    if text_path.exists() and not force:
        print(f"  ↷  Skipping (already downloaded): {work['title']}")
        return True

    print(f"  ↓  {work['author']}: {work['title']} ({work['year']})")
    index_url = work["url"]

    # ── Fetch the index / TOC page ───────────────────────────────────────────
    time.sleep(cfg.REQUEST_DELAY_S)
    soup = fetch_html(session, index_url)
    if soup is None:
        return False

    # ── Collect chapter URLs (may just be the index page itself) ─────────────
    chapter_urls = find_chapter_links(soup, index_url)

    # If the index page has no chapter links it *is* the content page
    if not chapter_urls:
        chapter_urls = [index_url]

    # ── Scrape each chapter ──────────────────────────────────────────────────
    all_text_parts = []
    failed = 0

    for i, url in enumerate(chapter_urls):
        if url != index_url:          # already fetched index above
            time.sleep(cfg.REQUEST_DELAY_S)
            chapter_soup = fetch_html(session, url)
            if chapter_soup is None:
                failed += 1
                continue
        else:
            chapter_soup = soup

        chapter_text = extract_text(chapter_soup)
        if chapter_text.strip():
            all_text_parts.append(f"[Source: {url}]\n\n{chapter_text}")

    if not all_text_parts:
        print(f"  ✗  No text extracted for: {work['title']}")
        return False

    full_text = "\n\n" + ("─" * 80 + "\n\n").join(all_text_parts)

    # ── Write outputs ─────────────────────────────────────────────────────────
    text_path.write_text(full_text, encoding="utf-8")

    metadata = {
        "id": work["id"],
        "author": work["author"],
        "title": work["title"],
        "year": work["year"],
        "url": work["url"],
        "chapter_urls": chapter_urls,
        "chapters_scraped": len(all_text_parts),
        "chapters_failed": failed,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"     ✓  {len(all_text_parts)} chapter(s) saved "
        f"({text_path.stat().st_size // 1024} KB)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Marxist corpus from MIA")
    parser.add_argument("--author", help="Filter to one author (e.g. Marx)")
    parser.add_argument("--id",     help="Download a single work by its ID")
    parser.add_argument("--force",  action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    corpus = json.loads(Path(cfg.CORPUS_CONFIG).read_text(encoding="utf-8"))

    # Apply filters
    if args.id:
        corpus = [w for w in corpus if w["id"] == args.id]
        if not corpus:
            sys.exit(f"No work found with id '{args.id}'")
    elif args.author:
        corpus = [w for w in corpus if w["author"].lower() == args.author.lower()]
        if not corpus:
            sys.exit(f"No works found for author '{args.author}'")

    print(f"\nDownloading {len(corpus)} work(s)…\n")
    session = get_session()
    ok = 0
    for work in corpus:
        if download_work(session, work, force=args.force):
            ok += 1

    print(f"\nDone: {ok}/{len(corpus)} works downloaded successfully.")
    if ok < len(corpus):
        print("Some works failed — check the output above for details.")


if __name__ == "__main__":
    main()
