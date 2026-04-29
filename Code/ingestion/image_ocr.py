from __future__ import annotations

from pathlib import Path

from Code.core.schema import RAGChunk
from Code.ingestion.document_ocr import DocumentOCR


def load_image_ocr_chunks(
    image_path: str | Path,
    ocr: DocumentOCR | None = None,
    chunk_start_id: int = 0,
) -> list[RAGChunk]:
    """
    OCR a standalone image file and return normalized RAG chunks.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ocr = ocr or DocumentOCR(lang="en")

    chunks = ocr.ocr_image_file(
        image_path=path,
        source=path,
        source_name=path.name,
        page=None,
        modality="image_ocr",
        block_type="image_text",
        chunk_id=chunk_start_id,
    )

    for i, chunk in enumerate(chunks, start=chunk_start_id):
        chunk.chunk_id = i
        chunk.modality = "image_ocr"
        chunk.block_type = "image_text"

    return chunks