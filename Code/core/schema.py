from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RAGChunk:
    """
    Normalized text unit used across the whole pipeline.

    Every input type becomes RAGChunk:
    - text PDF page/chunk
    - OCR PDF page/chunk
    - image OCR chunk
    - video transcript segment
    - video frame OCR chunk
    """

    chunk_id: int
    text: str

    # Source identity
    source: str | None = None
    source_name: str | None = None
    modality: str = "text"

    # Document metadata
    page: int | None = None
    block_type: str = "text"

    # Video/audio metadata
    start: float | None = None
    end: float | None = None

    # Flexible extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGChunk":
        return cls(
            chunk_id=int(data.get("chunk_id", -1)),
            text=data.get("text", ""),
            source=data.get("source"),
            source_name=data.get("source_name"),
            modality=data.get("modality", "text"),
            page=data.get("page"),
            block_type=data.get("block_type", "text"),
            start=data.get("start"),
            end=data.get("end"),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class RetrievalResult:
    """
    A retrieved chunk plus its similarity score.
    """

    chunk: RAGChunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "chunk": self.chunk.to_dict(),
        }


@dataclass
class IndexedFile:
    """
    File-level manifest entry.

    This supports deterministic answers for:
    - what files are uploaded?
    - how many PDFs?
    - what sources are indexed?
    """

    path: str
    name: str
    extension: str
    file_type: str
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionResult:
    """
    Result returned by ingestion/router after processing files.
    """

    chunks: list[RAGChunk]
    files: list[IndexedFile]

    def total_chunks(self) -> int:
        return len(self.chunks)

    def total_files(self) -> int:
        return len(self.files)