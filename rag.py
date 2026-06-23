"""
rag.py
──────
RAG pipeline: loads the persisted ChromaDB index and exposes a query interface
used by the Streamlit app.
"""

import sys
from pathlib import Path
from typing import Optional

import chromadb
import config as cfg

try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.vector_stores.types import (
        MetadataFilter,
        MetadataFilters,
        FilterOperator,
        FilterCondition,
    )
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nRun:  pip install -r requirements.txt")


class AuthorDiversityPostprocessor(BaseNodePostprocessor):
    """
    Re-ranks a large candidate pool so that:
    1. Primary authors (Marx, Engels) appear first.
    2. Each author contributes at most one chunk in the first pass,
       ensuring breadth across sources.
    3. Remaining slots are filled with the highest-scoring leftover chunks.
    The final list is capped at `max_nodes`.
    """

    max_nodes: int = cfg.TOP_K
    primary_authors: list = cfg.PRIMARY_AUTHORS

    @classmethod
    def class_name(cls) -> str:
        return "AuthorDiversityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return nodes

        by_score = sorted(nodes, key=lambda n: n.score or 0, reverse=True)

        result: list[NodeWithScore] = []
        seen_authors: set[str] = set()
        used_ids: set[str] = set()

        # Pass 1 — best chunk from each primary author
        for node in by_score:
            author = node.node.metadata.get("author", "Unknown")
            if author in self.primary_authors and author not in seen_authors:
                result.append(node)
                seen_authors.add(author)
                used_ids.add(node.node.node_id)

        # Pass 2 — best chunk from each remaining (secondary) author
        for node in by_score:
            if node.node.node_id in used_ids:
                continue
            author = node.node.metadata.get("author", "Unknown")
            if author not in seen_authors:
                result.append(node)
                seen_authors.add(author)
                used_ids.add(node.node.node_id)

        # Pass 3 — fill remaining slots with highest-scoring unused chunks
        for node in by_score:
            if len(result) >= self.max_nodes:
                break
            if node.node.node_id not in used_ids:
                result.append(node)
                used_ids.add(node.node.node_id)

        return result[: self.max_nodes]


class MarxistRAG:
    """
    Wraps the LlamaIndex + ChromaDB + Ollama pipeline.
    Instantiate once (expensive) then call .query() many times.
    """

    def __init__(
        self,
        author_filter: Optional[list[str]] = None,
        llm_model: str = cfg.LLM_MODEL,
    ):
        self._author_filter = author_filter  # e.g. ["Marx", "Engels"]
        self._llm_model = llm_model
        self._index = None
        self._query_engine = None
        self._build()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Connect to ChromaDB and configure LlamaIndex."""
        # Embedding model (same one used during ingestion)
        embed_model = OllamaEmbedding(
            model_name=cfg.EMBEDDING_MODEL,
            base_url=cfg.OLLAMA_BASE_URL,
        )

        # LLM for generation
        llm = Ollama(
            model=self._llm_model,
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=cfg.LLM_TEMPERATURE,
            context_window=cfg.LLM_CONTEXT_WIN,
            request_timeout=180.0,
            system_prompt=cfg.SYSTEM_PROMPT,
        )

        Settings.embed_model = embed_model
        Settings.llm = llm

        # Load persisted ChromaDB
        client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
        collection = client.get_or_create_collection(
            cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

        self._query_engine = self._make_query_engine(llm)

    def _make_query_engine(self, llm, author_filter: Optional[list[str]] = None):
        """Build (or rebuild) the query engine, optionally filtering by author."""
        authors = author_filter or self._author_filter

        # Build LlamaIndex MetadataFilters for author restriction
        metadata_filters = None
        if authors:
            if len(authors) == 1:
                metadata_filters = MetadataFilters(
                    filters=[MetadataFilter(key="author", value=authors[0], operator=FilterOperator.EQ)]
                )
            else:
                metadata_filters = MetadataFilters(
                    filters=[
                        MetadataFilter(key="author", value=a, operator=FilterOperator.EQ)
                        for a in authors
                    ],
                    condition=FilterCondition.OR,
                )

        # Fetch a larger candidate pool so the diversity re-ranker has room to work
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=cfg.RETRIEVAL_CANDIDATES,
            filters=metadata_filters,
        )

        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.25),
                AuthorDiversityPostprocessor(max_nodes=cfg.TOP_K, primary_authors=cfg.PRIMARY_AUTHORS),
            ],
            response_mode="compact",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_author_filter(self, authors: Optional[list[str]]) -> None:
        """Rebuild the query engine with a new author filter."""
        self._author_filter = authors
        self._query_engine = self._make_query_engine(authors)

    def query(self, question: str) -> dict:
        """
        Run a RAG query. Returns:
            {
                "answer": str,
                "sources": [
                    {
                        "author": str,
                        "title": str,
                        "year": str,
                        "excerpt": str,
                        "url": str,
                        "score": float,
                    },
                    …
                ]
            }
        """
        response = self._query_engine.query(question)

        sources = []
        seen_excerpts = set()
        for node_with_score in (response.source_nodes or []):
            node = node_with_score.node
            meta = node.metadata or {}
            excerpt = node.get_content()[:400].strip()
            # Deduplicate near-identical excerpts
            key = excerpt[:80]
            if key in seen_excerpts:
                continue
            seen_excerpts.add(key)
            sources.append(
                {
                    "author":  meta.get("author", "Unknown"),
                    "title":   meta.get("title", "Unknown work"),
                    "year":    meta.get("year", ""),
                    "excerpt": excerpt,
                    "url":     meta.get("source_url", ""),
                    "score":   round(node_with_score.score or 0, 3),
                }
            )

        # Sort by relevance score descending
        sources.sort(key=lambda s: s["score"], reverse=True)

        return {
            "answer":  str(response),
            "sources": sources,
        }

    def chunk_count(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
            col = client.get_collection(cfg.CHROMA_COLLECTION)
            return col.count()
        except Exception:
            return 0
