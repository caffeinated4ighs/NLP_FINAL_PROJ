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