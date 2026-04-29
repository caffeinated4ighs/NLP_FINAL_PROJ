from __future__ import annotations

from pathlib import Path

import fitz

from Code.core.schema import RAGChunk


def load_pdf_text_chunks(
    pdf_path: str | Path,
    chunk_start_id: int = 0,
) -> list[RAGChunk]:
    """
    Extract text from a text-based PDF page by page.

    This does not OCR scanned/image-based PDFs.
    If a PDF has no extractable text, this returns an empty list.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    chunks: list[RAGChunk] = []

    doc = fitz.open(path)

    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            chunks.append(
                RAGChunk(
                    chunk_id=chunk_start_id + len(chunks),
                    text=text,
                    source=str(path),
                    source_name=path.name,
                    modality="pdf_text",
                    page=page_num,
                    block_type="page_text",
                    metadata={
                        "loader": "pymupdf",
                    },
                )
            )
    finally:
        doc.close()

    return chunks


def pdf_has_text(pdf_path: str | Path) -> bool:
    """
    Fast check for whether a PDF has extractable text.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)

    try:
        for page in doc:
            if page.get_text().strip():
                return True
    finally:
        doc.close()

    return False