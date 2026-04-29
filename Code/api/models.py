from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class UploadResponse(BaseModel):
    session_id: str
    uploaded_files: list[str]


class InitRequest(BaseModel):
    session_id: str
    use_ocr_for_all_pdfs: bool = False
    include_video_frame_ocr: bool = True
    video_frame_interval_sec: int = 10


class StatusResponse(BaseModel):
    session_id: str
    status: str
    ready: bool
    uploaded_files: list[str] = Field(default_factory=list)
    indexed_files: list[dict] = Field(default_factory=list)
    error: str | None = None


class AskRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    source_contains: str | None = None
    modality: str | None = None


class AskResponse(BaseModel):
    session_id: str
    answer: str


class SummaryRequest(BaseModel):
    session_id: str


class SummaryResponse(BaseModel):
    session_id: str
    summary: str


class QuizRequest(BaseModel):
    session_id: str
    num_questions: int = 10
    difficulty: str = "mixed"


class QuizResponse(BaseModel):
    session_id: str
    quiz: str


class FlashcardRequest(BaseModel):
    session_id: str
    num_cards: int = 12


class FlashcardResponse(BaseModel):
    session_id: str
    flashcards: str


class FilesResponse(BaseModel):
    session_id: str
    files_markdown: str


class SourcesResponse(BaseModel):
    session_id: str
    sources_markdown: str