# Marxist Knowledge-Base AI

A fully-local AI assistant trained on the primary works of **Marx, Engels, Lenin, Trotsky, Luxemburg, and Gramsci**. Ask it anything — it retrieves the most relevant passages from 50+ original texts and synthesises a grounded, cited answer.

**No cloud. No API keys. Runs entirely on your machine.**

---

## How it works

```
Your question
     │
     ▼
nomic-embed-text (Ollama)          ← turns your question into a vector
     │
     ▼
ChromaDB vector search             ← finds the 6 most relevant passages
     │
     ▼
llama3.1:8b (Ollama)               ← reads passages + your question → writes answer
     │
     ▼
Streamlit web UI                   ← displays answer + source citations
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | `python --version` to check |
| **Ollama** | Download from [ollama.com](https://ollama.com) |
| **~10 GB disk** | For models + corpus |
| **4–8 GB RAM/VRAM** | More = faster; works on CPU but slower |

---

## Setup (one-time)

### 1. Install Ollama

Windows: download the installer from [ollama.com/download](https://ollama.com/download)

Then pull the two models you need:

```powershell
ollama pull nomic-embed-text    # embedding model (~274 MB)
ollama pull qwen3.6:27b         # main language model (~17 GB) — needs ~20 GB RAM/VRAM
```

> **If you have a local Ollama model named `qwen3.5:9b`, select it in the app instead.**
>
> **Lower-end machine?** Replace `qwen3.6:27b` with `qwen3.5:9b`, `qwen3.5:27b`, `qwen3:27b`, `qwen3:8b`, `llama3.1:8b`, `mistral:latest`, or `gemma3:3b`. Edit `config.py` → `LLM_MODEL`.

### 2. Clone the repo & install Python dependencies

```powershell
git clone https://github.com/yourname/marx-knowledge-base
cd marx-knowledge-base

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
```

### 3. Download the corpus (~60 works from Marxists Internet Archive)

```powershell
python scripts/download_corpus.py
```

This takes **5–20 minutes** depending on your connection. It politely rate-limits requests to MIA (0.6 s between pages). All texts are saved to `data/raw/`.

You can download a single author first to test:

```powershell
python scripts/download_corpus.py --author Marx
```

### 4. Embed and index the corpus

Make sure Ollama is running (`ollama serve` in another terminal, or it may start automatically):

```powershell
python scripts/ingest.py
```

This embeds every text chunk with `nomic-embed-text` and stores vectors in `chroma_db/`. First run takes **20–90 minutes** depending on hardware. Subsequent runs skip already-indexed works.

### 5. Launch the app

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser.

---

## Usage

- Type any question in the chat box.
- The AI retrieves the most relevant passages from the corpus and writes an answer citing author + work.
- Use the **Author Filter** in the sidebar to restrict which thinkers are consulted.
- Expand **Sources** below any answer to see the exact passages retrieved and links back to MIA.
- Switch LLMs in the sidebar without restarting.

### Example questions

- *What is the Marxist theory of surplus value?*
- *How does Lenin's theory of imperialism relate to Marx's analysis of capital?*
- *What would a Marxist say about identity politics?*
- *What is Gramsci's concept of hegemony?*
- *How did Luxemburg's view of reform differ from Lenin's?*
- *What is the materialist conception of history?*

---

## Adding more works

Edit `corpus_config.json` and add an entry:

```json
{
  "id": "unique-slug",
  "author": "Author Name",
  "title": "Full Title",
  "year": 1917,
  "url": "https://www.marxists.org/archive/author/works/year/title/index.htm"
}
```

Then run:

```powershell
python scripts/download_corpus.py --id unique-slug
python scripts/ingest.py --id unique-slug
```

---

## Tuning

All settings are in `config.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `LLM_MODEL` | `llama3.1:8b` | LLM used for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K` | `6` | Passages retrieved per query |
| `CHUNK_SIZE` | `512` | Tokens per indexed chunk |
| `LLM_TEMPERATURE` | `0.3` | Lower = more faithful to sources |
| `LLM_CONTEXT_WIN` | `8192` | Must match your model's actual context |

---

## File structure

```
marx-knowledge-base/
├── app.py                  ← Streamlit web UI
├── rag.py                  ← RAG pipeline (LlamaIndex)
├── config.py               ← All settings
├── corpus_config.json      ← Curated list of works
├── requirements.txt
├── scripts/
│   ├── download_corpus.py  ← Scrapes MIA → data/raw/
│   └── ingest.py           ← Embeds texts → chroma_db/
├── data/raw/{Author}/      ← Downloaded .txt + .json per work
└── chroma_db/              ← ChromaDB vector store (created at ingest time)
```

---

## Legal / ethical notes

- All texts on [Marxists Internet Archive](https://www.marxists.org) are in the **public domain** or licensed **CC BY-SA**. This project downloads them for personal research use.
- The scraper respects a rate limit and identifies itself in its User-Agent string.
- No data is sent to any third party; everything runs locally.
