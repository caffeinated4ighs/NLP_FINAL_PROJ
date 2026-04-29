from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from Code.core.schema import RAGChunk, RetrievalResult


class FAISSVectorStore:
    """
    Local FAISS vector store with JSON chunk metadata persistence.

    This is free, local, and does not require AWS/IAM.
    """

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[RAGChunk] = []

    def build(
        self,
        chunks: list[RAGChunk],
        embeddings: np.ndarray,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks provided.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{embeddings.shape[0]} embeddings"
            )

        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunks = chunks

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if self.index is None:
            raise ValueError("FAISS index is empty. Build or load an index first.")

        if query_embedding.ndim != 2:
            raise ValueError(
                f"Expected query embedding shape (1, dim), got {query_embedding.shape}"
            )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype="float32"),
            k,
        )

        results: list[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(idx)],
                    score=float(score),
                )
            )

        return results

    def save(self, index_dir: str | Path) -> None:
        if self.index is None:
            raise ValueError("Cannot save empty FAISS index.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "index.faiss"))

        chunks_payload = [chunk.to_dict() for chunk in self.chunks]

        with (index_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunks_payload, f, indent=2)

    def load(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)

        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk metadata: {chunks_path}")

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks_payload = json.load(f)

        self.chunks = [RAGChunk.from_dict(item) for item in chunks_payload]

    def count(self) -> int:
        return len(self.chunks)