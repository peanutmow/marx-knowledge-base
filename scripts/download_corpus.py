"""
scripts/download_corpus.py
──────────────────────────
Interactive corpus downloader for the Marxist Knowledge Base.

Downloads works from the Marxists Internet Archive (marxists.org)
with optional package / author / category selection.

Usage:
    python scripts/download_corpus.py               # interactive menu
    python scripts/download_corpus.py --author Marx  # one author only
    python scripts/download_corpus.py --author Lenin --category major
    python scripts/download_corpus.py --id marx-capital-v1
    python scripts/download_corpus.py --package marx-essential
    python scripts/download_corpus.py --force        # re-download
    python scripts/download_corpus.py --list         # list available works
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata

# Force UTF-8 encoding for stdout/stderr (handles Unicode on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

KEEP_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li"}
STRIP_TAGS = {
    "script", "style", "nav", "header", "footer", "form",
    "button", "input", "select", "noscript",
}

# ── Pre-defined optional packages ─────────────────────────────────────────────
# Each package is a set of author + category filters.
# Packages can also specify individual work IDs.
PACKAGES = {
    "marx-essential": {
        "label": "Marx — Essential Works (16 works, priority 1)",
        "authors": ["Marx"],
        "categories": ["major"],
        "priority_max": 1,
    },
    "marx-comprehensive": {
        "label": "Marx — Comprehensive (50+ works, all categories)",
        "authors": ["Marx"],
        "categories": None,
        "priority_max": None,
    },
    "engels-essential": {
        "label": "Engels — Essential Works (10 works, priority 1)",
        "authors": ["Engels"],
        "categories": ["major"],
        "priority_max": 1,
    },
    "engels-comprehensive": {
        "label": "Engels — Comprehensive (25+ works, all categories)",
        "authors": ["Engels"],
        "categories": None,
        "priority_max": None,
    },
    "lenin-essential": {
        "label": "Lenin — Essential Works (10 works, priority 1)",
        "authors": ["Lenin"],
        "categories": ["major"],
        "priority_max": 1,
    },
    "lenin-comprehensive": {
        "label": "Lenin — Comprehensive (40+ works, all categories)",
        "authors": ["Lenin"],
        "categories": None,
        "priority_max": None,
    },
    "all-essential": {
        "label": "All Authors — Essential Works (priority 1 only)",
        "authors": None,
        "categories": ["major"],
        "priority_max": 1,
    },
    "all-marx-engels": {
        "label": "Marx & Engels — All Works",
        "authors": ["Marx", "Engels"],
        "categories": None,
        "priority_max": None,
    },
    "all-comprehensive": {
        "label": "Everything — Complete Corpus",
        "authors": None,
        "categories": None,
        "priority_max": None,
    },
}

CATEGORY_LABELS = {
    "major": "Major Works (books, foundational texts)",
    "early": "Early Writings (pre-1848)",
    "political": "Political Writings (speeches, addresses, articles)",
    "economic": "Economic Writings (manuscripts, theories)",
    "articles": "Newspaper Articles",
    "philosophy": "Philosophical Works",
    "collected": "Collected/Selected Works Volumes",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Scraping helpers  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
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
    try:
        r = session.get(url, timeout=cfg.REQUEST_TIMEOUT_S)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, "lxml")
    except Exception as exc:
        print(f"  [WARN]  Failed to fetch {url}: {exc}")
        return None


def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()
    lines = []
    for el in soup.find_all(True):
        if el.name not in KEEP_TAGS:
            continue
        text = el.get_text(separator=" ", strip=True)
        if len(text) < 15:
            continue
        text = re.sub(r"\s+", " ", text)
        lines.append(text)
    return "\n\n".join(lines)


def find_chapter_links(soup: BeautifulSoup, index_url: str) -> list[str]:
    base = index_url.rsplit("/", 1)[0] + "/"
    base_parsed = urlparse(base)
    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(index_url, href)
        parsed = urlparse(full)
        if parsed.netloc != "www.marxists.org":
            continue
        if not parsed.path.startswith(base_parsed.path):
            continue
        if not re.search(r"\.html?$", parsed.path, re.IGNORECASE):
            continue
        if full == index_url or full.rstrip("/") == index_url.rstrip("/"):
            continue
        clean = parsed._replace(fragment="").geturl()
        if clean not in seen:
            seen.add(clean)
            links.append(clean)
    return links


# ═════════════════════════════════════════════════════════════════════════════
#  Download logic (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════

def download_work(session: requests.Session, work: dict, force: bool = False) -> bool:
    author_dir = Path(cfg.DATA_DIR) / slugify(work["author"])
    author_dir.mkdir(parents=True, exist_ok=True)
    text_path = author_dir / f"{work['id']}.txt"
    meta_path = author_dir / f"{work['id']}.json"
    if text_path.exists() and not force:
        print(f"  [skip]  Skipping (already downloaded): {work['title']}")
        return True
    print(f"  [down]  {work['author']}: {work['title']} ({work['year']})")
    index_url = work["url"]
    time.sleep(cfg.REQUEST_DELAY_S)
    soup = fetch_html(session, index_url)
    if soup is None:
        return False
    chapter_urls = find_chapter_links(soup, index_url)
    if not chapter_urls:
        chapter_urls = [index_url]
    all_text_parts = []
    failed = 0
    for i, url in enumerate(chapter_urls):
        if url != index_url:
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
        print(f"  [X]  No text extracted for: {work['title']}")
        return False
    full_text = "\n\n" + ("─" * 80 + "\n\n").join(all_text_parts)
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
    print(f"     ✓  {len(all_text_parts)} chapter(s) saved ({text_path.stat().st_size // 1024} KB)")
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Interactive menu
# ═════════════════════════════════════════════════════════════════════════════

def run_interactive(corpus: list[dict]) -> list[dict]:
    """Present an interactive menu for selecting works to download."""
    authors = sorted(set(w["author"] for w in corpus))
    categories = sorted(set(w.get("category", "other") for w in corpus))

    print("\n" + "=" * 72)
    print("  MARXIST KNOWLEDGE BASE — Corpus Downloader")
    print("=" * 72)

    # ── Step 1: Choose a pre-defined package or custom selection ──────────
    print("\n[Pkg]   Select an optional package (or choose custom selection):")
    print("─" * 72)
    package_keys = sorted(PACKAGES.keys())
    for i, key in enumerate(package_keys, 1):
        pkg = PACKAGES[key]
        # Count works matching this package
        filtered = _filter_by_package(corpus, pkg)
        print(f"  [{i:>2}] {pkg['label']}  ({len(filtered)} works)")
    print(f"  [{len(package_keys) + 1:>2}] Custom selection (pick author + categories)")
    print()

    while True:
        try:
            choice = input("  Enter choice [1–{}], or 'q' to quit: ".format(len(package_keys) + 1)).strip()
            if choice.lower() in ("q", "quit", "exit"):
                print("  Exiting.")
                sys.exit(0)
            choice_num = int(choice)
            if 1 <= choice_num <= len(package_keys):
                pkg = PACKAGES[package_keys[choice_num - 1]]
                return _filter_by_package(corpus, pkg)
            elif choice_num == len(package_keys) + 1:
                return _custom_selection(corpus, authors, categories)
            else:
                print(f"  Please enter a number between 1 and {len(package_keys) + 1}.")
        except ValueError:
            print("  Please enter a valid number.")

    return corpus


def _filter_by_package(corpus: list[dict], pkg: dict) -> list[dict]:
    """Filter corpus according to a package definition."""
    result = corpus
    if pkg["authors"] is not None:
        result = [w for w in result if w["author"] in pkg["authors"]]
    if pkg["categories"] is not None:
        result = [w for w in result if w.get("category") in pkg["categories"]]
    if pkg.get("priority_max") is not None:
        result = [w for w in result if w.get("priority", 99) <= pkg["priority_max"]]
    return result


def _custom_selection(corpus: list[dict], authors: list[str], categories: list[str]) -> list[dict]:
    """Let user pick authors and categories interactively."""
    selected = corpus[:]

    # ── Step 2: Pick authors ──────────────────────────────────────────────
    print("\n👤  Select authors (comma-separated, or 'all'):")
    print("─" * 72)
    for i, a in enumerate(authors, 1):
        count = sum(1 for w in corpus if w["author"] == a)
        print(f"  [{i}] {a} ({count} works)")
    print()
    author_input = input("  Authors (e.g. '1,2' or 'all'): ").strip()
    if author_input.lower() not in ("all", ""):
        try:
            indices = [int(x.strip()) for x in author_input.split(",")]
            chosen_authors = [authors[i - 1] for i in indices if 1 <= i <= len(authors)]
            selected = [w for w in selected if w["author"] in chosen_authors]
        except (ValueError, IndexError):
            print("  Invalid selection, using all authors.")

    # ── Step 3: Pick categories ───────────────────────────────────────────
    print("\n📂  Select categories (comma-separated, or 'all'):")
    print("─" * 72)
    valid_cats = sorted(c for c in categories if c != "other")
    for i, cat in enumerate(valid_cats, 1):
        label = CATEGORY_LABELS.get(cat, cat)
        count = sum(1 for w in selected if w.get("category") == cat)
        print(f"  [{i}] {label} ({count} works)")
    print()
    cat_input = input("  Categories (e.g. '1,3' or 'all'): ").strip()
    if cat_input.lower() not in ("all", ""):
        try:
            indices = [int(x.strip()) for x in cat_input.split(",")]
            chosen_cats = [valid_cats[i - 1] for i in indices if 1 <= i <= len(valid_cats)]
            selected = [w for w in selected if w.get("category") in chosen_cats]
        except (ValueError, IndexError):
            print("  Invalid selection, using all categories.")

    # ── Step 4: Priority filter ───────────────────────────────────────────
    print("\n[P]  Filter by priority:")
    print("  [1] Only priority 1 (essential works)")
    print("  [2] Priority 1 + 2 (essential + recommended)")
    print("  [3] All priorities (everything)")
    print()
    prio_input = input("  Choose [1–3] (default 3): ").strip()
    if prio_input == "1":
        selected = [w for w in selected if w.get("priority", 99) == 1]
    elif prio_input == "2":
        selected = [w for w in selected if w.get("priority", 99) <= 2]

    return selected


# ═════════════════════════════════════════════════════════════════════════════
#  Summary display
# ═════════════════════════════════════════════════════════════════════════════

def show_summary(selected: list[dict]):
    """Print a summary of the works that will be downloaded."""
    by_author = defaultdict(list)
    for w in selected:
        by_author[w["author"]].append(w)

    print("\n" + "─" * 72)
    print(f"  [Sum]   Summary: {len(selected)} work(s) to download")
    print("─" * 72)
    for author in sorted(by_author.keys()):
        works = sorted(by_author[author], key=lambda w: (w.get("priority", 99), w.get("year", 0)))
        print(f"\n  👤  {author} ({len(works)} works):")
        for w in works:
            cat = w.get("category", "")
            prio = w.get("priority", "")
            print(f"      {'*' if prio == 1 else 'o'} {w['title']} ({w['year']})  [{cat}]")
    print()


def confirm_download() -> bool:
    """Ask user to confirm before downloading."""
    while True:
        resp = input("  Proceed with download? [Y/n]: ").strip().lower()
        if resp in ("", "y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download Marxist corpus from MIA with optional package selection",
    )
    parser.add_argument("--author", help="Filter to one author (e.g. Marx)")
    parser.add_argument("--id", help="Download a single work by its ID")
    parser.add_argument("--category", help="Filter by category (e.g. major, early, political)")
    parser.add_argument("--package", help="Use a pre-defined package (see --list-packages)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--list", action="store_true", help="List all works and exit")
    parser.add_argument("--list-packages", action="store_true", help="List available packages and exit")
    parser.add_argument("--noninteractive", action="store_true",
                        help="Skip interactive menu; use filters or download all")
    args = parser.parse_args()

    # ── Load corpus ─────────────────────────────────────────────────────────
    corpus = json.loads(Path(cfg.CORPUS_CONFIG).read_text(encoding="utf-8"))

    # ── List mode ───────────────────────────────────────────────────────────
    if args.list:
        print(f"\n[Books]  Complete Corpus ({len(corpus)} works):\n")
        by_author = defaultdict(list)
        for w in corpus:
            by_author[w["author"]].append(w)
        for author in sorted(by_author.keys()):
            print(f"  {author}:")
            for w in sorted(by_author[author], key=lambda x: (x.get("priority", 99), x.get("year", 0))):
                cat = w.get("category", "")
                prio = w.get("priority", "")
                print(f"    {'*' if prio == 1 else 'o'} {w['id']:40s} {w['title']:60s} ({w['year']})  [{cat}]")
            print()
        return

    if args.list_packages:
        print(f"\n[Pkg]   Available Optional Packages:\n")
        for key, pkg in sorted(PACKAGES.items()):
            filtered = _filter_by_package(corpus, pkg)
            print(f"  {key}")
            print(f"      {pkg['label']}")
            print(f"      → {len(filtered)} works")
            print()
        return

    # ── Determine which works to download ────────────────────────────────────
    if args.id:
        selected = [w for w in corpus if w["id"] == args.id]
        if not selected:
            sys.exit(f"No work found with id '{args.id}'")
    elif args.package:
        if args.package not in PACKAGES:
            sys.exit(f"Unknown package '{args.package}'. Use --list-packages to see available packages.")
        selected = _filter_by_package(corpus, PACKAGES[args.package])
        print(f"\n[Pkg]   Package '{args.package}': {len(selected)} works selected.")
    elif args.author or args.category or args.noninteractive:
        selected = corpus[:]
        if args.author:
            selected = [w for w in selected if w["author"].lower() == args.author.lower()]
        if args.category:
            selected = [w for w in selected if w.get("category") == args.category]
        if not selected:
            sys.exit("No works match the given filters.")
    else:
        # Interactive mode
        selected = run_interactive(corpus)

    if not selected:
        print("No works selected. Exiting.")
        return

    # ── Show summary and confirm ────────────────────────────────────────────
    show_summary(selected)

    # In interactive mode, ask for confirmation
    if not args.id and not args.noninteractive:
        if not confirm_download():
            print("Download cancelled.")
            return

    # ── Download ─────────────────────────────────────────────────────────────
    print(f"\nDownloading {len(selected)} work(s)…\n")
    session = get_session()
    ok = 0
    for work in selected:
        if download_work(session, work, force=args.force):
            ok += 1

    print(f"\nDone: {ok}/{len(selected)} works downloaded successfully.")
    if ok < len(selected):
        print("Some works failed — check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
