from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGChunk:
    chunk_id: int
    text: str
    source: str | None = None
    modality: str = "text"

    # PDF / document fields
    page: int | None = None
    block_type: str = "text"

    # Video/audio fields
    start: float | None = None
    end: float | None = None

    # Flexible extras: OCR boxes, confidence, image path, table info, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: RAGChunk
    score: float