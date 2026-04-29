from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from paddleocr import PaddleOCR

from Code.core.schema import RAGChunk
from Code.core.settings import PDF_RENDER_DPI, PROCESSED_DIR


class DocumentOCR:
    """
    OCR engine for scanned PDFs and rendered PDF pages.

    Output is normalized to RAGChunk so downstream code does not need to
    special-case dictionaries.
    """

    def __init__(self, lang: str = "en", show_log: bool = False):
        self.lang = lang
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=show_log,
        )

    def pdf_to_images(
        self,
        pdf_path: str | Path,
        out_dir: str | Path | None = None,
        dpi: int = PDF_RENDER_DPI,
    ) -> list[Path]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if out_dir is None:
            out_dir = PROCESSED_DIR / "pdf_pages" / pdf_path.stem

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        image_paths: list[Path] = []

        try:
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=matrix)

                image_path = out_dir / f"{pdf_path.stem}_page_{page_idx + 1}.png"
                pix.save(str(image_path))
                image_paths.append(image_path)
        finally:
            doc.close()

        return image_paths

    def ocr_image_file(
        self,
        image_path: str | Path,
        source: str | Path | None = None,
        source_name: str | None = None,
        page: int | None = None,
        modality: str = "pdf_ocr",
        block_type: str = "page_ocr",
        chunk_id: int = -1,
    ) -> list[RAGChunk]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        source_str = str(source) if source is not None else str(image_path)
        source_name = source_name or Path(source_str).name

        result = self.ocr.ocr(str(image_path), cls=True)

        if not result or not result[0]:
            return []

        lines: list[str] = []
        boxes: list[dict[str, Any]] = []

        for item in result[0]:
            bbox = item[0]
            text = item[1][0]
            confidence = float(item[1][1])

            if not text or not text.strip():
                continue

            lines.append(text.strip())
            boxes.append(
                {
                    "bbox": bbox,
                    "text": text.strip(),
                    "confidence": confidence,
                }
            )

        full_text = "\n".join(lines).strip()

        if not full_text:
            return []

        return [
            RAGChunk(
                chunk_id=chunk_id,
                text=full_text,
                source=source_str,
                source_name=source_name,
                modality=modality,
                page=page,
                block_type=block_type,
                metadata={
                    "ocr_engine": "paddleocr",
                    "lang": self.lang,
                    "image_path": str(image_path),
                    "boxes": boxes,
                },
            )
        ]

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        work_dir: str | Path | None = None,
    ) -> list[RAGChunk]:
        pdf_path = Path(pdf_path)

        page_images = self.pdf_to_images(
            pdf_path=pdf_path,
            out_dir=work_dir,
        )

        chunks: list[RAGChunk] = []

        for page_num, image_path in enumerate(page_images, start=1):
            page_chunks = self.ocr_image_file(
                image_path=image_path,
                source=pdf_path,
                source_name=pdf_path.name,
                page=page_num,
                modality="pdf_ocr",
                block_type="page_ocr",
            )
            chunks.extend(page_chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i

        return chunks