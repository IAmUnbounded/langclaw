"""Persistent memory module — long-term recall using vector search.

Provides semantic memory storage and retrieval using ChromaDB.
Falls back to a simple JSON-based store if ChromaDB is not installed.

Storage location: ~/.openclaw/memory/
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".openclaw" / "memory"


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


class MemoryStore:
    """Persistent memory store with semantic search.

    Uses ChromaDB for vector search if available,
    otherwise falls back to keyword-based search over a JSON file.
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
