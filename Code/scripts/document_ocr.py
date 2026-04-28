from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR


class DocumentOCR:
    def __init__(self, lang: str = "en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def pdf_to_images(self, pdf_path: str, out_dir: str, dpi: int = 200) -> List[str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        image_paths = []

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img_path = out / f"{Path(pdf_path).stem}_page_{page_idx + 1}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))

        return image_paths

    def ocr_image(self, image_path: str, source: str, page: int | None = None) -> List[Dict[str, Any]]:
        result = self.ocr.ocr(image_path, cls=True)
        chunks = []

        if not result or not result[0]:
            return chunks

        lines = []
        boxes = []

        for item in result[0]:
            bbox = item[0]
            text = item[1][0]
            confidence = item[1][1]

            if text.strip():
                lines.append(text)
                boxes.append({"bbox": bbox, "text": text, "confidence": confidence})

        full_text = "\n".join(lines).strip()

        if full_text:
            chunks.append({
                "source": source,
                "modality": "ocr",
                "page": page,
                "block_type": "page_text",
                "text": full_text,
                "metadata": {
                    "image_path": image_path,
                    "ocr_engine": "paddleocr",
                    "boxes": boxes,
                }
            })

        return chunks

    def ocr_pdf(self, pdf_path: str, work_dir: str = "data/processed/pdf_pages") -> List[Dict[str, Any]]:
        image_paths = self.pdf_to_images(pdf_path, work_dir)
        all_chunks = []

        for i, image_path in enumerate(image_paths, start=1):
            all_chunks.extend(
                self.ocr_image(
                    image_path=image_path,
                    source=pdf_path,
                    page=i,
                )
            )

        return all_chunks