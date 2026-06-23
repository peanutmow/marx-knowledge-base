"""
scripts/ingest.py
─────────────────
Chunks all downloaded texts, embeds them via Ollama, and stores them in ChromaDB.

Usage:
    python scripts/ingest.py              # ingest everything not yet ingested
    python scripts/ingest.py --force      # re-embed even already-indexed works
    python scripts/ingest.py --author Marx
    python scripts/ingest.py --id marx-capital-v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 encoding for stdout/stderr (handles Unicode on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import chromadb

# ── allow running from project root ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

# LlamaIndex imports (lazy so error messages are clear)
try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Run:  pip install -r requirements.txt"
    )


def check_ollama_model(model_name: str) -> None:
    """Verify the embedding model is available in Ollama."""
    import urllib.request
    import urllib.error
    url = f"{cfg.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        names = [m["name"] for m in data.get("models", [])]
        # model names in Ollama can be "nomic-embed-text:latest" etc.
        base_names = [n.split(":")[0] for n in names]
        if model_name not in names and model_name not in base_names:
            print(
                f"⚠  Ollama model '{model_name}' not found.\n"
                f"   Run:  ollama pull {model_name}\n"
                f"   Found: {', '.join(names) or '(none)'}"
            )
    except Exception as exc:
        print(f"⚠  Could not reach Ollama at {cfg.OLLAMA_BASE_URL}: {exc}")
        print("   Make sure `ollama serve` is running.")


def build_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=cfg.EMBEDDING_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        embed_batch_size=4,          # conservative for local hardware
    )


def setup_chroma() -> tuple[chromadb.Collection, ChromaVectorStore]:
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    collection = client.get_or_create_collection(
        cfg.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    store = ChromaVectorStore(chroma_collection=collection)
    return collection, store


def already_ingested(collection: chromadb.Collection, work_id: str) -> bool:
    """Check if any chunk for this work is already in ChromaDB."""
    results = collection.get(
        where={"work_id": work_id},
        limit=1,
    )
    return len(results["ids"]) > 0


def load_work_documents(meta_path: Path, text_path: Path) -> list[Document]:
    """Load a single work into LlamaIndex Document objects (one per chapter section)."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    full_text = text_path.read_text(encoding="utf-8")

    # Split on the section separator written by the scraper
    sections = full_text.split("─" * 80)
    documents = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract the [Source: url] header if present
        source_url = meta["url"]
        if section.startswith("[Source:"):
            first_line_end = section.index("]") + 1
            source_url = section[len("[Source:"):first_line_end - 1].strip()
            section = section[first_line_end:].strip()

        if len(section) < 100:   # too short to be useful
            continue

        doc = Document(
            text=section,
            metadata={
                "work_id":    meta["id"],
                "author":     meta["author"],
                "title":      meta["title"],
                "year":       str(meta["year"]),
                "source_url": source_url,
            },
        )
        documents.append(doc)

    return documents


def ingest_work(
    work_id: str,
    collection: chromadb.Collection,
    vector_store: ChromaVectorStore,
    embed_model: OllamaEmbedding,
    force: bool,
) -> bool:
    """Find the .txt + .json pair for work_id, chunk it, embed, and store."""
    # Find the files
    data_root = Path(cfg.DATA_DIR)
    text_files = list(data_root.glob(f"**/{work_id}.txt"))
    meta_files = list(data_root.glob(f"**/{work_id}.json"))

    if not text_files or not meta_files:
        print(f"  ✗  No downloaded files found for work id '{work_id}'. Run the scraper first.")
        return False

    text_path = text_files[0]
    meta_path = meta_files[0]

    # Load metadata to get title for display
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label = f"{meta['author']}: {meta['title']} ({meta['year']})"

    if not force and already_ingested(collection, work_id):
        print(f"  ↷  Already indexed: {label}")
        return True

    if force:
        # Delete existing chunks for this work
        existing = collection.get(where={"work_id": work_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    print(f"  ↓  Ingesting: {label}")
    documents = load_work_documents(meta_path, text_path)
    if not documents:
        print(f"  ✗  No usable text sections found.")
        return False

    # Chunk
    splitter = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"     {len(documents)} section(s) → {len(nodes)} chunk(s)")

    # Set up LlamaIndex storage context pointing at ChromaDB
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = embed_model
    Settings.llm = None       # no LLM needed during ingestion

    # Build/update the index (this triggers embedding + upsert)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Embed and store Marxist corpus in ChromaDB")
    parser.add_argument("--author", help="Filter to one author (e.g. Marx)")
    parser.add_argument("--id",     help="Ingest a single work by its ID")
    parser.add_argument("--force",  action="store_true", help="Re-embed already-indexed works")
    args = parser.parse_args()

    # ── Load corpus manifest ─────────────────────────────────────────────────
    corpus = json.loads(Path(cfg.CORPUS_CONFIG).read_text(encoding="utf-8"))
    if args.id:
        corpus = [w for w in corpus if w["id"] == args.id]
        if not corpus:
            sys.exit(f"No work found with id '{args.id}'")
    elif args.author:
        corpus = [w for w in corpus if w["author"].lower() == args.author.lower()]

    # ── Only attempt works that have been downloaded ─────────────────────────
    data_root = Path(cfg.DATA_DIR)
    available = {p.stem for p in data_root.glob("**/*.txt")}
    to_ingest = [w for w in corpus if w["id"] in available]
    skipped_missing = len(corpus) - len(to_ingest)

    if skipped_missing:
        print(
            f"ℹ  {skipped_missing} work(s) not yet downloaded "
            f"— run `python scripts/download_corpus.py` first."
        )

    if not to_ingest:
        sys.exit("Nothing to ingest.")

    print(f"\nIngesting {len(to_ingest)} work(s)…\n")

    # ── Setup ────────────────────────────────────────────────────────────────
    check_ollama_model(cfg.EMBEDDING_MODEL)
    embed_model = build_embed_model()
    collection, vector_store = setup_chroma()

    ok = 0
    for work in to_ingest:
        if ingest_work(work["id"], collection, vector_store, embed_model, args.force):
            ok += 1

    total_chunks = collection.count()
    print(f"\nDone: {ok}/{len(to_ingest)} works ingested.")
    print(f"ChromaDB collection '{cfg.CHROMA_COLLECTION}' now holds {total_chunks:,} chunks.")


if __name__ == "__main__":
    main()
