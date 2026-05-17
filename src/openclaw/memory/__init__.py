"""Persistent memory module — long-term recall using vector search.

Provides semantic memory storage and retrieval using ChromaDB.
Falls back to a simple JSON-based store if ChromaDB is not installed.

Includes hybrid search combining vector similarity with BM25 keyword ranking,
optional MMR (Maximal Marginal Relevance) reranking for diversity, and
optional temporal decay to prefer recent memories.

Storage location: ~/.langclaw/memory/
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".langclaw" / "memory"


@dataclass
class MemoryEntry:
    """A single memory entry."""

    memory_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", 0.0),
        )


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Simple BM25 ranking implementation for keyword-based search.

    Parameters follow the standard BM25 formulation:
    - k1 controls term-frequency saturation (default 1.5)
    - b controls document-length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freqs: Counter[str] = Counter()
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def index(self, documents: list[str]) -> None:
        """Build the BM25 index from a list of document strings."""
        self._doc_tokens = [_tokenize(doc) for doc in documents]
        self._doc_lens = [len(tokens) for tokens in self._doc_tokens]
        self._n_docs = len(documents)
        self._avg_dl = (
            sum(self._doc_lens) / self._n_docs if self._n_docs > 0 else 1.0
        )

        self._doc_freqs = Counter()
        for tokens in self._doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        df = self._doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF with smoothing
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Score all indexed documents against the query.

        Returns a list of BM25 scores, one per document, in index order.
        """
        query_tokens = _tokenize(query)
        scores = [0.0] * self._n_docs

        for q_term in query_tokens:
            idf = self._idf(q_term)
            if idf <= 0:
                continue
            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(q_term)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self._avg_dl
                )
                scores[i] += idf * (numerator / denominator)

        return scores


class MemoryStore:
    """Persistent memory store with semantic search.

    Uses ChromaDB for vector search if available,
    otherwise falls back to keyword-based search over a JSON file.

    Supports hybrid search combining vector similarity with BM25 keyword ranking,
    optional MMR reranking for result diversity, and temporal decay.
    """

    def __init__(self, memory_dir: Path | None = None):
        self._dir = memory_dir or MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._chroma_client = None
        self._collection = None
        self._fallback_file = self._dir / "memories.jsonl"

        # Try to initialize ChromaDB
        try:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(
                path=str(self._dir / "chromadb")
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="openclaw_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Memory store: using ChromaDB vector search")
        except ImportError:
            logger.info("Memory store: ChromaDB not installed, using JSON fallback")
        except Exception as e:
            logger.warning(f"Memory store: ChromaDB init failed ({e}), using JSON fallback")

    def save(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Save a memory entry."""
        entry = MemoryEntry(
            memory_id=str(uuid.uuid4())[:12],
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
        )

        if self._collection is not None:
            self._collection.add(
                ids=[entry.memory_id],
                documents=[entry.content],
                metadatas=[{**entry.metadata, "timestamp": str(entry.timestamp)}],
            )
        else:
            # JSON fallback
            with open(self._fallback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

        logger.info(f"Saved memory {entry.memory_id}: {content[:50]}...")
        return entry

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self._collection.count() or 1),
                )
                entries = []
                if results and results["ids"] and results["ids"][0]:
                    for i, mid in enumerate(results["ids"][0]):
                        doc = results["documents"][0][i] if results["documents"] else ""
                        meta = results["metadatas"][0][i] if results["metadatas"] else {}
                        ts = float(meta.pop("timestamp", 0))
                        entries.append(MemoryEntry(
                            memory_id=mid,
                            content=doc,
                            metadata=meta,
                            timestamp=ts,
                        ))
                return entries
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {e}")
                return []
        else:
            # Keyword fallback
            return self._keyword_search(query, top_k)

    def list_recent(self, n: int = 10) -> list[MemoryEntry]:
        """List the most recent memories."""
        if self._collection is not None:
            try:
                count = self._collection.count()
                if count == 0:
                    return []
                results = self._collection.get(
                    limit=min(n, count),
                )
                entries = []
                if results and results["ids"]:
                    for i, mid in enumerate(results["ids"]):
                        doc = results["documents"][i] if results["documents"] else ""
                        meta = results["metadatas"][i] if results["metadatas"] else {}
                        ts = float(meta.pop("timestamp", 0))
                        entries.append(MemoryEntry(
                            memory_id=mid,
                            content=doc,
                            metadata=meta,
                            timestamp=ts,
                        ))
                entries.sort(key=lambda e: e.timestamp, reverse=True)
                return entries[:n]
            except Exception as e:
                logger.warning(f"ChromaDB list failed: {e}")
                return []
        else:
            return self._load_all_fallback()[-n:]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if self._collection is not None:
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False
        else:
            entries = self._load_all_fallback()
            filtered = [e for e in entries if e.memory_id != memory_id]
            if len(filtered) == len(entries):
                return False
            self._save_all_fallback(filtered)
            return True

    def _keyword_search(self, query: str, top_k: int) -> list[MemoryEntry]:
        """Simple keyword-based search fallback."""
        entries = self._load_all_fallback()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for entry in entries:
            content_lower = entry.content.lower()
            # Score by word overlap
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0 or query_lower in content_lower:
                score = overlap + (1 if query_lower in content_lower else 0)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _load_all_fallback(self) -> list[MemoryEntry]:
        """Load all memories from the JSON fallback file."""
        if not self._fallback_file.exists():
            return []
        entries = []
        for line in self._fallback_file.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(MemoryEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    pass
        return entries

    def _save_all_fallback(self, entries: list[MemoryEntry]) -> None:
        """Save all memories to the JSON fallback file."""
        with open(self._fallback_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Hybrid search: vector + BM25 with optional MMR and temporal decay
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        temporal_decay: bool = False,
        half_life_days: float = 30.0,
    ) -> list[MemoryEntry]:
        """Search memories using a hybrid of vector similarity and BM25 keyword ranking.

        Scores from each method are min-max normalized to [0, 1] and combined
        with the given weights. Optionally applies MMR reranking for diversity
        and/or temporal decay to prefer recent memories.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.
            vector_weight: Weight for vector similarity scores (default 0.7).
            text_weight: Weight for BM25 text scores (default 0.3).
            use_mmr: If True, apply MMR reranking for diversity.
            mmr_lambda: MMR trade-off between relevance and diversity (0-1).
            temporal_decay: If True, apply exponential time-decay to scores.
            half_life_days: Half-life in days for temporal decay.

        Returns:
            List of MemoryEntry objects ranked by combined score.
        """
        # Fetch a larger candidate pool for reranking
        candidate_k = max(top_k * 3, 20)

        # --- Vector scores ---------------------------------------------------
        vector_entries: list[MemoryEntry] = []
        vector_scores: list[float] = []

        if self._collection is not None:
            try:
                count = self._collection.count()
                if count > 0:
                    n = min(candidate_k, count)
                    results = self._collection.query(
                        query_texts=[query],
                        n_results=n,
                        include=["documents", "metadatas", "distances"],
                    )
                    if results and results["ids"] and results["ids"][0]:
                        for i, mid in enumerate(results["ids"][0]):
                            doc = results["documents"][0][i] if results["documents"] else ""
                            meta = dict(results["metadatas"][0][i]) if results["metadatas"] else {}
                            ts = float(meta.pop("timestamp", 0))
                            entry = MemoryEntry(
                                memory_id=mid,
                                content=doc,
                                metadata=meta,
                                timestamp=ts,
                            )
                            vector_entries.append(entry)
                            # ChromaDB cosine distance: lower = more similar.
                            # Convert to similarity: similarity = 1 - distance
                            dist = results["distances"][0][i] if results.get("distances") else 0.0
                            vector_scores.append(1.0 - dist)
            except Exception as e:
                logger.warning(f"ChromaDB hybrid vector search failed: {e}")
        else:
            # Fallback: no vector scores available; use all entries
            vector_entries = self._load_all_fallback()
            vector_scores = [0.0] * len(vector_entries)

        if not vector_entries:
            return []

        # --- BM25 scores ------------------------------------------------------
        documents = [e.content for e in vector_entries]
        bm25 = BM25()
        bm25.index(documents)
        bm25_scores = bm25.score(query)

        # --- Normalize scores to [0, 1] --------------------------------------
        norm_vector = self._min_max_normalize(vector_scores)
        norm_bm25 = self._min_max_normalize(bm25_scores)

        # --- Combine scores ---------------------------------------------------
        combined: list[tuple[float, int]] = []
        for i in range(len(vector_entries)):
            score = vector_weight * norm_vector[i] + text_weight * norm_bm25[i]
            combined.append((score, i))

        # --- Temporal decay (applied before MMR) ------------------------------
        if temporal_decay:
            now = time.time()
            ln2 = math.log(2)
            half_life_seconds = half_life_days * 86400.0
            decayed = []
            for score, idx in combined:
                age_seconds = max(now - vector_entries[idx].timestamp, 0)
                decay = math.exp(-ln2 * age_seconds / half_life_seconds)
                decayed.append((score * decay, idx))
            combined = decayed

        # --- Sort by combined score descending --------------------------------
        combined.sort(key=lambda x: x[0], reverse=True)

        if use_mmr:
            return self._mmr_rerank(
                combined, vector_entries, top_k, mmr_lambda
            )

        return [vector_entries[idx] for _, idx in combined[:top_k]]

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        """Normalize a list of scores to [0, 1] using min-max scaling."""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span == 0:
            return [0.0] * len(scores)
        return [(s - min_s) / span for s in scores]

    @staticmethod
    def _mmr_rerank(
        scored: list[tuple[float, int]],
        entries: list[MemoryEntry],
        top_k: int,
        mmr_lambda: float,
    ) -> list[MemoryEntry]:
        """Apply Maximal Marginal Relevance reranking for diversity.

        Uses token-overlap (Jaccard similarity) between documents as the
        diversity measure, avoiding the need for embedding vectors.

        Args:
            scored: List of (combined_score, index) tuples, sorted descending.
            entries: The full list of MemoryEntry candidates.
            top_k: Number of results to select.
            mmr_lambda: Trade-off between relevance (1.0) and diversity (0.0).

        Returns:
            List of top_k MemoryEntry objects reranked by MMR.
        """
        if not scored:
            return []

        # Pre-tokenize all candidate documents for similarity computation
        candidate_tokens: dict[int, set[str]] = {}
        for _, idx in scored:
            candidate_tokens[idx] = set(_tokenize(entries[idx].content))

        # Build a score lookup
        score_map = {idx: sc for sc, idx in scored}

        selected_indices: list[int] = []
        remaining = {idx for _, idx in scored}

        for _ in range(min(top_k, len(scored))):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                relevance = score_map[idx]

                # Max similarity to already-selected documents
                if selected_indices:
                    max_sim = max(
                        _jaccard(candidate_tokens[idx], candidate_tokens[s])
                        for s in selected_indices
                    )
                else:
                    max_sim = 0.0

                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx < 0:
                break

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return [entries[idx] for idx in selected_indices]


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
