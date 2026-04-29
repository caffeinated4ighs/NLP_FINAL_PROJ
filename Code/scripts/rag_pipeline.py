from pathlib import Path
from collections import defaultdict

import fitz

from Code.scripts.schema import RAGChunk
from Code.scripts.embedder import BGERAGEmbedder
from Code.scripts.orchestrator import QwenOrchestrator

try:
    from Code.scripts.vector_store import ChromaBGEVectorStore
except ImportError:
    ChromaBGEVectorStore = None


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        top_k: int = 5,
        use_4bit: bool = True,
        prompt_config_path: str = "Code/configs/prompts.yaml",
        use_chroma: bool = False,
        chroma_persist_dir: str = "data/vector_store/chroma",
    ):
        self.top_k = top_k
        self.use_chroma = use_chroma
        self.file_manifest = []

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

        if self.use_chroma:
            if ChromaBGEVectorStore is None:
                raise ImportError(
                    "use_chroma=True, but Code.scripts.vector_store.ChromaBGEVectorStore "
                    "could not be imported. Either add vector_store.py or set use_chroma=False."
                )

            self.vector_store = ChromaBGEVectorStore(
                embedding_model_name=embedding_model_name,
                persist_dir=chroma_persist_dir,
                top_k=top_k,
            )
        else:
            self.vector_store = None

    # ---------------------------------------------------------------------
    # Chunk / manifest helpers
    # ---------------------------------------------------------------------

    def _normalize_chunk(self, chunk, chunk_id: int) -> RAGChunk:
        if isinstance(chunk, dict):
            return RAGChunk(
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                source=chunk.get("source"),
                modality=chunk.get("modality", "unknown"),
                page=chunk.get("page"),
                block_type=chunk.get("block_type", "text"),
                start=chunk.get("start"),
                end=chunk.get("end"),
                metadata=chunk.get("metadata", {}),
            )

        chunk.chunk_id = chunk_id
        return chunk

    def _normalize_chunks(self, chunks: list) -> list[RAGChunk]:
        normalized_chunks = []

        for i, chunk in enumerate(chunks):
            normalized = self._normalize_chunk(chunk, i)

            if normalized.text and normalized.text.strip():
                normalized_chunks.append(normalized)

        for i, chunk in enumerate(normalized_chunks):
            chunk.chunk_id = i

        return normalized_chunks

    def _add_manifest_entry(self, file_path, ext: str, modality: str) -> None:
        self.file_manifest.append(
            {
                "path": str(file_path),
                "extension": ext,
                "modality": modality,
            }
        )

    def list_uploaded_files(self) -> str:
        if not self.file_manifest:
            return "No files have been indexed yet."

        lines = [
            "| # | File | Type | Modality |",
            "|---:|---|---|---|",
        ]

        for i, item in enumerate(self.file_manifest, start=1):
            lines.append(
                f"| {i} | {item['path']} | {item['extension']} | {item['modality']} |"
            )

        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # PDF loading
    # ---------------------------------------------------------------------

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

        doc.close()
        return pages

    # ---------------------------------------------------------------------
    # Index builders
    # ---------------------------------------------------------------------

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

        normalized_chunks = self._normalize_chunks(all_chunks)

        print(f"Created {len(normalized_chunks)} chunks.")
        print("Creating embeddings and building index...")

        self.build_index_from_chunks(normalized_chunks)

        print("Index ready.")

    def build_index_from_chunks(self, chunks: list[RAGChunk]) -> None:
        normalized_chunks = self._normalize_chunks(chunks)

        # Keep chunks in memory even when Chroma is enabled.
        # This lets manifest/source summary/per-source quiz prep work.
        self.embedder.chunks = normalized_chunks

        if self.use_chroma:
            self.vector_store.add_chunks(normalized_chunks, reset=True)
        else:
            self.embedder.build_index_from_chunks(normalized_chunks)

    def build_index_from_data_folder(
        self,
        data_dir: str = "data",
        use_ocr_for_all_pdfs: bool = False,
        video_frame_interval_sec: int = 10,
    ) -> None:
        from Code.scripts.document_ocr import DocumentOCR
        from Code.scripts.video_ingest import VideoIngestor

        data_path = Path(data_dir)
        self.file_manifest = []

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
            p
            for p in data_path.rglob("*")
            if (
                p.is_file()
                and not p.name.startswith(".")
                and data_path / "processed" not in p.parents
                and data_path / "vector_store" not in p.parents
            )
        )

        if not files:
            raise ValueError(f"No files found in {data_path}")

        for file_path in files:
            ext = file_path.suffix.lower()
            file_str = str(file_path)

            print(f"\nProcessing: {file_str}")

            if ext in pdf_exts:
                self._add_manifest_entry(file_path, ext, "pdf")

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

                if use_ocr_for_all_pdfs or text_chunk_count == 0:
                    print(f"Running OCR for PDF: {file_str}")
                    ocr_chunks = ocr.ocr_pdf(
                        pdf_path=file_str,
                        work_dir="data/processed/pdf_pages",
                    )
                    all_chunks.extend(ocr_chunks)

            elif ext in image_exts:
                self._add_manifest_entry(file_path, ext, "image_ocr")

                print(f"Running OCR for image: {file_str}")
                image_chunks = ocr.ocr_image(
                    image_path=file_str,
                    source=file_str,
                    page=None,
                )
                all_chunks.extend(image_chunks)

            elif ext in video_exts:
                self._add_manifest_entry(file_path, ext, "video")

                print(f"Processing video: {file_str}")
                video_chunks = video_ingestor.process_video(file_str)
                all_chunks.extend(video_chunks)

            else:
                print(f"Skipping unsupported file type: {file_str}")

        if not all_chunks:
            raise ValueError("No chunks created from data folder.")

        normalized_chunks = self._normalize_chunks(all_chunks)

        print(f"\nTotal chunks created: {len(normalized_chunks)}")
        self.build_index_from_chunks(normalized_chunks)

        print("\nIndex ready.")

    # ---------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None):
        if self.use_chroma:
            return self.vector_store.retrieve(query, top_k=top_k or self.top_k)

        return self.embedder.retrieve(query, top_k=top_k)

    def retrieve_filtered(
        self,
        query: str,
        top_k: int = 10,
        source_contains: str | None = None,
        modality: str | None = None,
    ):
        if self.use_chroma:
            # Chroma can do exact metadata filters. For source substring,
            # we still do candidate filtering in Python.
            candidates = self.vector_store.retrieve(
                query=query,
                top_k=max(50, top_k * 5),
            )
        else:
            candidates = self.embedder.retrieve(
                query=query,
                top_k=max(50, top_k * 5),
            )

        filtered = []

        for result in candidates:
            chunk = result.chunk

            if source_contains:
                if not chunk.source or source_contains.lower() not in chunk.source.lower():
                    continue

            if modality:
                if chunk.modality != modality:
                    continue

            filtered.append(result)

        return filtered[:top_k]

    # ---------------------------------------------------------------------
    # Answering
    # ---------------------------------------------------------------------

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)
        return self.orchestrator.answer(question, retrieved)

    def answer_filtered(
        self,
        question: str,
        source_contains: str | None = None,
        modality: str | None = None,
        top_k: int = 10,
    ) -> str:
        retrieved = self.retrieve_filtered(
            query=question,
            source_contains=source_contains,
            modality=modality,
            top_k=top_k,
        )

        if not retrieved:
            return "I could not find matching indexed content for that filter."

        return self.orchestrator.answer(question, retrieved)

    def summarize(self) -> str:
        retrieved = self.retrieve("Summarize the main topics in this coursework.")
        return self.orchestrator.summarize(retrieved)

    def generate_quiz(self) -> str:
        retrieved = self.retrieve("Generate a quiz from this coursework.")
        return self.orchestrator.generate_quiz(retrieved)

    # ---------------------------------------------------------------------
    # Metadata/source utilities
    # ---------------------------------------------------------------------

    def get_source_summary_table(self) -> str:
        if not self.embedder.chunks:
            return "No chunks indexed."

        sources = {}

        for chunk in self.embedder.chunks:
            source = chunk.source or "unknown"

            if source not in sources:
                sources[source] = {
                    "chunks": 0,
                    "modalities": set(),
                    "pages": set(),
                    "timestamps": 0,
                }

            sources[source]["chunks"] += 1
            sources[source]["modalities"].add(chunk.modality)

            if chunk.page is not None:
                sources[source]["pages"].add(chunk.page)

            if chunk.start is not None or chunk.end is not None:
                sources[source]["timestamps"] += 1

        lines = [
            "| Source | Chunks | Modalities | Pages | Timed chunks |",
            "|---|---:|---|---:|---:|",
        ]

        for source, info in sorted(sources.items()):
            pages = len(info["pages"]) if info["pages"] else 0
            modalities = ", ".join(sorted(info["modalities"]))

            lines.append(
                f"| {source} | {info['chunks']} | {modalities} | {pages} | {info['timestamps']} |"
            )

        return "\n".join(lines)

    def summarize_each_source(self, max_chunks_per_source: int = 8) -> str:
        if not self.embedder.chunks:
            return "No chunks indexed."

        sources = defaultdict(list)

        for chunk in self.embedder.chunks:
            source = chunk.source or "unknown"
            sources[source].append(chunk)

        outputs = []

        for source, chunks in sorted(sources.items()):
            sample_chunks = chunks[:max_chunks_per_source]

            context = "\n\n".join(
                f"[SOURCE: {source} | CHUNK: {chunk.chunk_id} | "
                f"MODALITY: {chunk.modality} | PAGE: {chunk.page} | "
                f"START: {chunk.start} | END: {chunk.end}]\n{chunk.text}"
                for chunk in sample_chunks
            )

            system_prompt = (
                "You are a quiz-prep assistant. "
                "Use only the provided source context. "
                "Return a concise markdown table."
            )

            user_prompt = f"""
Source:
{source}

Context:
{context}

Break this source down for quiz preparation.

Return this exact table format:

| Source | Main topics | Key terms | Quiz prep notes |
|---|---|---|---|
"""

            summary = self.orchestrator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=450,
                benchmark=True,
            )

            outputs.append(summary)

        return "\n\n".join(outputs)