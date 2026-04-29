from __future__ import annotations

from pathlib import Path

from Code.core.schema import IndexedFile, IngestionResult, RAGChunk
from Code.core.settings import PROCESSED_DIR
from Code.ingestion.document_ocr import DocumentOCR
from Code.ingestion.image_ocr import load_image_ocr_chunks
from Code.ingestion.pdf_loader import load_pdf_text_chunks
from Code.ingestion.video_ingest import VideoIngestor


PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
TEXT_EXTS = {".txt", ".md"}


def iter_supported_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        return [path]

    ignored_dirs = {
        "processed",
        "indexes",
        "vector_store",
        "__pycache__",
        ".git",
        ".venv",
    }

    files = []

    for item in path.rglob("*"):
        if not item.is_file():
            continue

        if item.name.startswith("."):
            continue

        if any(part in ignored_dirs for part in item.parts):
            continue

        ext = item.suffix.lower()

        if ext in PDF_EXTS or ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext in TEXT_EXTS:
            files.append(item)

    return sorted(files)


def load_text_file_chunks(
    path: str | Path,
    chunk_start_id: int = 0,
) -> list[RAGChunk]:
    path = Path(path)

    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        RAGChunk(
            chunk_id=chunk_start_id,
            text=text,
            source=str(path),
            source_name=path.name,
            modality="text",
            block_type="text_file",
            metadata={"loader": "text_file"},
        )
    ]


def ingest_files(
    input_path: str | Path,
    use_ocr_for_all_pdfs: bool = False,
    include_video_frame_ocr: bool = True,
    video_frame_interval_sec: int = 10,
) -> IngestionResult:
    files = iter_supported_files(input_path)

    if not files:
        raise ValueError(f"No supported files found in: {input_path}")

    ocr = DocumentOCR(lang="en")
    video_ingestor = VideoIngestor(
        frame_interval_sec=video_frame_interval_sec,
        ocr=ocr,
    )

    all_chunks: list[RAGChunk] = []
    manifest: list[IndexedFile] = []

    for file_path in files:
        ext = file_path.suffix.lower()
        chunk_start_id = len(all_chunks)

        print(f"\nProcessing: {file_path}")

        if ext in PDF_EXTS:
            chunks = load_pdf_text_chunks(
                pdf_path=file_path,
                chunk_start_id=chunk_start_id,
            )

            if use_ocr_for_all_pdfs or not chunks:
                print(f"Running OCR for PDF: {file_path}")
                chunks = ocr.ocr_pdf(
                    pdf_path=file_path,
                    work_dir=PROCESSED_DIR / "pdf_pages" / file_path.stem,
                )

                for i, chunk in enumerate(chunks, start=chunk_start_id):
                    chunk.chunk_id = i

            file_type = "pdf"

        elif ext in IMAGE_EXTS:
            chunks = load_image_ocr_chunks(
                image_path=file_path,
                ocr=ocr,
                chunk_start_id=chunk_start_id,
            )
            file_type = "image"

        elif ext in VIDEO_EXTS:
            chunks = video_ingestor.process_video(
                video_path=file_path,
                chunk_start_id=chunk_start_id,
                include_frame_ocr=include_video_frame_ocr,
            )
            file_type = "video"

        elif ext in TEXT_EXTS:
            chunks = load_text_file_chunks(
                path=file_path,
                chunk_start_id=chunk_start_id,
            )
            file_type = "text"

        else:
            continue

        all_chunks.extend(chunks)

        manifest.append(
            IndexedFile(
                path=str(file_path),
                name=file_path.name,
                extension=ext,
                file_type=file_type,
                chunk_count=len(chunks),
            )
        )

    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i

    return IngestionResult(
        chunks=all_chunks,
        files=manifest,
    )