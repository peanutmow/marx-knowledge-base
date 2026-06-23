# Marxist Knowledge-Base AI

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Ollama" src="https://img.shields.io/badge/ollama-local-orange">
</p>

A fully-local AI assistant trained on over **125+ primary works** of **Marx, Engels, and Lenin**. Retrieves the most relevant passages from original texts and synthesises a grounded, cited answer.
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
qwen3.5:9b (Ollama)                ← reads passages + your question → writes answer
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
ollama pull qwen3.5:9b          # main language model (~5.5 GB)
```

> **Higher-end machine?** Replace `qwen3.5:9b` with `qwen3.6:27b` (~17 GB), `qwen3:27b`, or `llama3.1:70b` for deeper reasoning. Lower-end: `qwen3:8b`, `llama3.1:8b`, `mistral:latest`, or `gemma3:3b`. Edit `config.py` → `LLM_MODEL`.

### 2. Clone the repo & install Python dependencies

```powershell
git clone https://github.com/peanutmow/marx-knowledge-base
cd marx-knowledge-base

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
```

### 3. Download the corpus (~125 works from Marxists Internet Archive)

The downloader offers **two modes**: interactive and CLI.

#### Interactive mode (recommended)

```powershell
python scripts/download_corpus.py
```

This presents a menu where you can choose from **pre-built optional packages**:

| Package | Works | Description |
|---------|-------|-------------|
| `marx-essential` | 16 | Marx's major works (Capital, Manifesto, Grundrisse, etc.) |
| `marx-comprehensive` | 53 | All Marx works including early writings, articles, political texts |
| `engels-essential` | 7 | Engels' major works (Anti-Dühring, Condition of Working Class, etc.) |
| `engels-comprehensive` | 32 | All Engels works including political writings, articles |
| `lenin-essential` | 9 | Lenin's major works (State & Revolution, Imperialism, etc.) |
| `lenin-comprehensive` | 40 | All Lenin works including congress speeches, articles |
| `all-essential` | 32 | All authors' priority-1 works |
| `all-marx-engels` | 85 | Everything by Marx and Engels |
| `all-comprehensive` | 125 | The complete corpus |

You can also create a **custom selection** by picking authors, categories, and priority levels.

#### CLI mode

```powershell
# Download a pre-defined package
python scripts/download_corpus.py --package marx-essential

# Download everything (non-interactive)
python scripts/download_corpus.py --noninteractive

# Single author
python scripts/download_corpus.py --author Marx

# Single work
python scripts/download_corpus.py --id marx-capital-v1

# List available works
python scripts/download_corpus.py --list

# List available packages
python scripts/download_corpus.py --list-packages
```

The download takes **5–20 minutes** depending on your connection. It politely rate-limits requests to MIA (0.6 s between pages). All texts are saved to `data/raw/`.

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
| `LLM_MODEL` | `qwen3.5:9b` | LLM used for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K` | `6` | Passages retrieved per query |
| `CHUNK_SIZE` | `512` | Tokens per indexed chunk |
| `LLM_TEMPERATURE` | `0.3` | Lower = more faithful to sources |
| `LLM_CONTEXT_WIN` | `32768` | Must match your model's actual context |

---

## File structure

```
marx-knowledge-base/
├── app.py                  ← Streamlit web UI
├── rag.py                  ← RAG pipeline (LlamaIndex)
├── config.py               ← All settings
├── corpus_config.json      ← Curated list of works
├── requirements.txt
├── LICENSE                 ← MIT license
├── .gitignore
├── scripts/
│   ├── download_corpus.py  ← Scrapes MIA → data/raw/
│   └── ingest.py           ← Embeds texts → chroma_db/
├── data/raw/{Author}/      ← Downloaded .txt + .json per work
└── chroma_db/              ← ChromaDB vector store (created at ingest time)
```

---

## License

The **source code** in this repository is licensed under the **MIT License** — see [`LICENSE`](LICENSE).

The **corpus texts** downloaded from [Marxists Internet Archive](https://www.marxists.org) are in the **public domain** or licensed **CC BY-SA**. They are downloaded for personal research use only.

## Ethical notes

- The scraper respects a rate limit and identifies itself in its User-Agent string.
- No data is sent to any third party; everything runs locally.
