from pathlib import Path

import fitz

from Code.scripts.schema import RAGChunk
from Code.scripts.embedder import BGERAGEmbedder
from Code.scripts.orchestrator import QwenOrchestrator


class RAGPipeline:
    # Code/scripts/rag_pipeline.py

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        top_k: int = 5,
        use_4bit: bool = True,
        prompt_config_path: str = "Code/configs/prompts.yaml",
    ):
        self.embedder = BGERAGEmbedder(
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )

        self.orchestrator = QwenOrchestrator(
            llm_model_name=llm_model_name,
            use_4bit=use_4bit,
            prompt_config_path=prompt_config_path,
        )



    def build_index_from_data_folder(
        self,
        data_dir: str = "data",
        use_ocr_for_all_pdfs: bool = False,
        video_frame_interval_sec: int = 10,
    ) -> None:
        from pathlib import Path

        from Code.scripts.document_ocr import DocumentOCR
        from Code.scripts.video_ingest import VideoIngestor

        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_path}")

        pdf_exts = {".pdf"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

        all_chunks = []

        ocr = DocumentOCR(lang="en")

        video_ingestor = VideoIngestor(
            whisper_model="base",
            device="cuda",
            compute_type="float16",
            frame_interval_sec=video_frame_interval_sec,
        )

        files = sorted(
            p for p in data_path.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        )

        if not files:
            raise ValueError(f"No files found in {data_path}")

        for file_path in files:
            ext = file_path.suffix.lower()
            file_str = str(file_path)

            print(f"\nProcessing: {file_str}")

            if ext in pdf_exts:
                # First try normal text extraction.
                pages = self.load_pdf_text_by_page(file_str)

                text_chunk_count = 0

                for page in pages:
                    page_chunks = self.embedder.chunk_text(
                        text=page["text"],
                        source=page["source"],
                        modality="pdf_text",
                        page=page["page"],
                        block_type="text",
                    )
                    all_chunks.extend(page_chunks)
                    text_chunk_count += len(page_chunks)

                # If PDF has no text, or user forces OCR, OCR the PDF pages.
                if use_ocr_for_all_pdfs or text_chunk_count == 0:
                    print(f"Running OCR for PDF: {file_str}")
                    ocr_chunks = ocr.ocr_pdf(
                        pdf_path=file_str,
                        work_dir="data/processed/pdf_pages",
                    )
                    all_chunks.extend(ocr_chunks)

            elif ext in image_exts:
                print(f"Running OCR for image: {file_str}")
                image_chunks = ocr.ocr_image(
                    image_path=file_str,
                    source=file_str,
                    page=None,
                )
                all_chunks.extend(image_chunks)

            elif ext in video_exts:
                print(f"Processing video: {file_str}")
                video_chunks = video_ingestor.process_video(file_str)
                all_chunks.extend(video_chunks)

            else:
                print(f"Skipping unsupported file type: {file_str}")

        if not all_chunks:
            raise ValueError("No chunks created from data folder.")

        # Reassign global chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i

        print(f"\nTotal chunks created: {len(all_chunks)}")
        self.build_index_from_chunks(all_chunks)
        
    def load_pdf_text_by_page(self, pdf_path: str) -> list[dict]:
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"File not found: {pdf_path_obj}")

        doc = fitz.open(pdf_path_obj)

        pages = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()

            if text.strip():
                pages.append(
                    {
                        "source": str(pdf_path_obj),
                        "page": page_num,
                        "text": text,
                    }
                )

        return pages

    def build_index(self, pdf_path: str) -> None:
        print(f"Loading PDF: {pdf_path}")

        pages = self.load_pdf_text_by_page(pdf_path)
        all_chunks: list[RAGChunk] = []

        print("Chunking text...")

        for page in pages:
            page_chunks = self.embedder.chunk_text(
                text=page["text"],
                source=page["source"],
                modality="pdf_text",
                page=page["page"],
                block_type="text",
            )
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError("No chunks created from PDF.")

        # Make chunk IDs globally unique after page-level chunking
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i

        print(f"Created {len(all_chunks)} chunks.")

        print("Creating embeddings and building FAISS index...")
        self.embedder.build_index_from_chunks(all_chunks)

        print("Index ready.")

    def build_index_from_chunks(self, chunks: list[RAGChunk]) -> None:
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i

        self.embedder.build_index_from_chunks(chunks)

    def retrieve(self, query: str):
        return self.embedder.retrieve(query)

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)
        return self.orchestrator.answer(question, retrieved)

    def summarize(self) -> str:
        retrieved = self.retrieve("Summarize the main topics in this coursework.")
        return self.orchestrator.summarize(retrieved)

    def generate_quiz(self) -> str:
        retrieved = self.retrieve("Generate a quiz from this coursework.")
        return self.orchestrator.generate_quiz(retrieved)