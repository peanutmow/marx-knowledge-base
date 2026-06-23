"""
app.py
──────
Streamlit chat UI for the Marxist Knowledge-Base AI.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

import config as cfg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marxist Knowledge Base",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_rag(llm_model: str, author_filter_key: str):
    """
    Cache the RAG pipeline. Re-instantiates when model or author filter changes.
    author_filter_key is a stringified list used as a hashable cache key.
    """
    from rag import MarxistRAG
    authors = json.loads(author_filter_key) if author_filter_key != "[]" else None
    return MarxistRAG(author_filter=authors, llm_model=llm_model)


def get_all_authors() -> list[str]:
    corpus = json.loads(Path(cfg.CORPUS_CONFIG).read_text(encoding="utf-8"))
    return sorted(set(w["author"] for w in corpus))


def corpus_stats() -> dict:
    """Return basic stats about downloaded + indexed works."""
    corpus = json.loads(Path(cfg.CORPUS_CONFIG).read_text(encoding="utf-8"))
    data_root = Path(cfg.DATA_DIR)
    downloaded = {p.stem for p in data_root.glob("**/*.txt")}
    total = len(corpus)
    done = sum(1 for w in corpus if w["id"] in downloaded)
    return {"total": total, "downloaded": done}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✊ Marxist Knowledge Base")
    st.markdown("*Powered by LlamaIndex · ChromaDB · Ollama*")
    st.divider()

    # Model selector
    st.subheader("Model")
    llm_options = [
        "qwen3.6:27b",
        "qwen3.5:27b",
        "qwen3.5:9b",
        "qwen3:27b",
        "qwen3:8b",
        "llama3.1:8b",
        "mistral:latest",
        "gemma3:3b",
        "qwen2.5:7b",
        "llama3.1:70b",
    ]
    llm_model = st.selectbox(
        "LLM",
        llm_options,
        index=llm_options.index(cfg.LLM_MODEL) if cfg.LLM_MODEL in llm_options else 0,
        help="Make sure the chosen model is pulled in Ollama.",
    )

    st.divider()

    # Author filter
    st.subheader("Author Filter")
    all_authors = get_all_authors()
    selected_authors = st.multiselect(
        "Include sources from",
        options=all_authors,
        default=all_authors,
        help="Deselect authors to exclude their works from retrieval.",
    )
    # Treat 'all selected' as no filter (faster)
    active_filter = selected_authors if len(selected_authors) < len(all_authors) else []
    author_filter_key = json.dumps(sorted(active_filter))

    st.divider()

    # Citation toggle
    show_sources = st.toggle("Show sources", value=True)

    st.divider()

    # Corpus status
    stats = corpus_stats()
    st.subheader("Corpus Status")
    st.metric("Works in corpus", stats["total"])
    st.metric("Works downloaded", stats["downloaded"])
    if stats["downloaded"] < stats["total"]:
        st.warning(
            f"{stats['total'] - stats['downloaded']} work(s) not yet downloaded.\n\n"
            "Run:\n```\npython scripts/download_corpus.py\npython scripts/ingest.py\n```"
        )

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.header("Ask the Marxist Knowledge Base")

# Show a warning if nothing is ingested yet
try:
    from rag import MarxistRAG
    import chromadb
    chroma_client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    try:
        col = chroma_client.get_collection(cfg.CHROMA_COLLECTION)
        chunk_count = col.count()
    except Exception:
        chunk_count = 0
except Exception:
    chunk_count = 0

if chunk_count == 0:
    st.warning(
        "The knowledge base is empty. You need to download and ingest the corpus first.\n\n"
        "**Setup steps:**\n"
        "```bash\n"
        "# 1. Pull Ollama models (run once)\n"
        "ollama pull nomic-embed-text\n"
        f"ollama pull {cfg.LLM_MODEL}\n\n"
        "# 2. Download texts from Marxists Internet Archive\n"
        "python scripts/download_corpus.py\n\n"
        "# 3. Embed and store in ChromaDB\n"
        "python scripts/ingest.py\n"
        "```"
    )
else:
    st.caption(f"Knowledge base loaded · {chunk_count:,} indexed chunks · {stats['downloaded']} works")

# Initialise conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    score_pct = int(src["score"] * 100)
                    st.markdown(
                        f"**{src['author']}** — *{src['title']}* ({src['year']}) "
                        f"· relevance: {score_pct}%"
                    )
                    if src["url"]:
                        st.markdown(f"[View on Marxists Internet Archive]({src['url']})")
                    st.markdown(f"> {src['excerpt'][:300]}…")
                    st.divider()

# ── Accept new input ──────────────────────────────────────────────────────────

if chunk_count == 0:
    st.chat_input("Knowledge base not yet loaded — see setup instructions above.", disabled=True)
else:
    if prompt := st.chat_input("Ask a question about Marxist theory…"):
        # Save + display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Load RAG (cached) and run query
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating response…"):
                try:
                    rag = load_rag(llm_model, author_filter_key)
                    result = rag.query(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as exc:
                    answer = (
                        f"**Error:** {exc}\n\n"
                        "Make sure Ollama is running (`ollama serve`) "
                        f"and the model `{llm_model}` is pulled."
                    )
                    sources = []

            st.markdown(answer)

            if show_sources and sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for src in sources:
                        score_pct = int(src["score"] * 100)
                        st.markdown(
                            f"**{src['author']}** — *{src['title']}* ({src['year']}) "
                            f"· relevance: {score_pct}%"
                        )
                        if src["url"]:
                            st.markdown(f"[View on Marxists Internet Archive]({src['url']})")
                        st.markdown(f"> {src['excerpt'][:300]}…")
                        st.divider()

        # Persist to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
