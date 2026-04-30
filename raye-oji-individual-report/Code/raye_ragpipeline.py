from __future__ import annotations

from pathlib import Path

from Code.core.schema import IndexedFile, IngestionResult, RAGChunk
from Code.core.settings import TOP_K
from Code.ingestion.router import ingest_files
from Code.rag.orchestrator import QwenOrchestrator
from Code.rag.quiz_tools import QuizTools
from Code.rag.retriever import Retriever


class RAGPipeline:
    """
    Main multimodal RAG coordinator.

    This class intentionally does not implement low-level OCR, video, embedding,
    or FAISS logic directly. It wires together the ingestion, retrieval,
    orchestration, and quiz/study tools.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        load_llm: bool = True,
    ) -> None:
        self.top_k = top_k

        self.retriever = Retriever(top_k=top_k)

        self.orchestrator: QwenOrchestrator | None = None
        self.quiz_tools: QuizTools | None = None

        if load_llm:
            self.load_llm()

        self.indexed_files: list[IndexedFile] = []

    @property
    def chunks(self) -> list[RAGChunk]:
        return self.retriever.chunks

    def load_llm(self) -> None:
        if self.orchestrator is None:
            self.orchestrator = QwenOrchestrator()
            self.quiz_tools = QuizTools(self.orchestrator)

    def _require_llm(self) -> QwenOrchestrator:
        if self.orchestrator is None:
            self.load_llm()

        if self.orchestrator is None:
            raise RuntimeError("LLM could not be loaded.")

        return self.orchestrator

    def _require_quiz_tools(self) -> QuizTools:
        self._require_llm()

        if self.quiz_tools is None:
            raise RuntimeError("Quiz tools are unavailable.")

        return self.quiz_tools

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index_from_path(
        self,
        input_path: str | Path,
        use_ocr_for_all_pdfs: bool = False,
        include_video_frame_ocr: bool = True,
        video_frame_interval_sec: int = 10,
    ) -> IngestionResult:
        result = ingest_files(
            input_path=input_path,
            use_ocr_for_all_pdfs=use_ocr_for_all_pdfs,
            include_video_frame_ocr=include_video_frame_ocr,
            video_frame_interval_sec=video_frame_interval_sec,
        )

        self.build_index_from_chunks(
            chunks=result.chunks,
            files=result.files,
        )

        return result

    def build_index_from_chunks(
        self,
        chunks: list[RAGChunk],
        files: list[IndexedFile] | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks provided.")

        self.retriever.build(
            chunks=chunks,
            files=files,
        )

        self.indexed_files = files or []

    # ------------------------------------------------------------------
    # Metadata utilities
    # ------------------------------------------------------------------

    def list_files(self) -> str:
        return self.retriever.list_files_table()

    def list_sources(self) -> str:
        return self.retriever.source_summary_table()

    # ------------------------------------------------------------------
    # Retrieval-backed generation
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> str:
        orchestrator = self._require_llm()
        retrieved = self.retriever.retrieve(question, top_k=top_k or self.top_k)
        return orchestrator.answer(question, retrieved)

    def answer_filtered(
        self,
        question: str,
        source_contains: str | None = None,
        modality: str | None = None,
        top_k: int | None = None,
    ) -> str:
        orchestrator = self._require_llm()

        retrieved = self.retriever.retrieve_filtered(
            query=question,
            top_k=top_k or self.top_k,
            source_contains=source_contains,
            modality=modality,
        )

        if not retrieved:
            return "I could not find matching indexed content for that filter."

        return orchestrator.answer(question, retrieved)

    def summarize(
        self,
        top_k: int | None = None,
    ) -> str:
        orchestrator = self._require_llm()

        retrieved = self.retriever.retrieve(
            "Summarize the main topics in this coursework.",
            top_k=top_k or self.top_k,
        )

        return orchestrator.summarize(retrieved)

    def generate_quiz(
        self,
        num_questions: int = 10,
        difficulty: str = "mixed",
    ) -> str:
        quiz_tools = self._require_quiz_tools()

        return quiz_tools.generate_quiz_from_chunks(
            chunks=self.chunks,
            num_questions=num_questions,
            difficulty=difficulty,
        )

    def generate_exam_questions(
        self,
        num_questions: int = 6,
        difficulty: str = "mixed",
    ) -> str:
        quiz_tools = self._require_quiz_tools()

        return quiz_tools.generate_exam_questions(
            chunks=self.chunks,
            num_questions=num_questions,
            difficulty=difficulty,
        )

    def generate_flashcards(
        self,
        num_cards: int = 12,
    ) -> str:
        quiz_tools = self._require_quiz_tools()

        return quiz_tools.generate_flashcards_from_chunks(
            chunks=self.chunks,
            num_cards=num_cards,
        )

    def quiz_prep_by_source(self) -> str:
        quiz_tools = self._require_quiz_tools()

        return quiz_tools.summarize_each_source(
            chunks=self.chunks,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, index_dir: str | Path) -> None:
        self.retriever.save(str(index_dir))

    def load_index(self, index_dir: str | Path) -> None:
        self.retriever.load(str(index_dir))