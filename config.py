"""
Central configuration for the Marxist Knowledge-Base AI.
Edit this file to change models, paths, or chunking behaviour.
"""
import os

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL",        "qwen3.5:9b")   # change to qwen3.6:27b, qwen3.5:9b, qwen3:27b, qwen3:8b, llama3.1:8b, mistral:latest, or gemma3:3b for lower-end hardware
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "nomic-embed-text")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data", "raw")
CHROMA_DIR        = os.path.join(BASE_DIR, "chroma_db")
CORPUS_CONFIG     = os.path.join(BASE_DIR, "corpus_config.json")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "marxist_works"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512    # tokens
CHUNK_OVERLAP     = 50     # tokens

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K                = 6    # final number of chunks passed to the LLM
RETRIEVAL_CANDIDATES = 20   # larger pool fetched from ChromaDB before re-ranking
PRIMARY_AUTHORS      = ["Marx", "Engels"]  # boosted to front in diversity re-ranking

# ── Scraper ───────────────────────────────────────────────────────────────────
REQUEST_DELAY_S   = 0.6    # seconds between HTTP requests (be polite to MIA)
REQUEST_TIMEOUT_S = 20     # seconds before giving up on a page

# ── LLM generation ────────────────────────────────────────────────────────────
LLM_TEMPERATURE   = 0.3    # lower = more faithful to sources
LLM_CONTEXT_WIN   = 32768  # tokens; qwen3:27b supports up to 32k (131k with extra config)

SYSTEM_PROMPT = """\
You are a scholarly assistant specialising in Marxist theory and political economy.
You have access to the primary works of Marx, Engels, Lenin, Trotsky, Luxemburg, Gramsci,
and other classical Marxist thinkers.

When answering questions:
1. Ground every claim in the retrieved source passages provided to you.
2. Always cite the author and work when referencing a specific idea (e.g. "As Marx argues in Capital…").
3. Treat Marx and Engels as the primary theoretical sources. When their works contain a clear
   answer, lead with their definition or analysis before discussing how later thinkers built on it.
4. If different thinkers hold different views on a topic, present each perspective fairly and note
   how secondary theorists (Lenin, Trotsky, Luxemburg, Gramsci) extended or departed from Marx.
5. Interpret questions through the framework of historical materialism, dialectics, and class analysis.
6. If the retrieved passages do not contain enough information to answer well, say so clearly rather
   than speculating beyond the sources.
7. Write clearly for an educated non-specialist reader.
"""
