from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from Code.api.models import (
    AskRequest,
    AskResponse,
    FilesResponse,
    FlashcardRequest,
    FlashcardResponse,
    HealthResponse,
    InitRequest,
    QuizRequest,
    QuizResponse,
    SourcesResponse,
    StatusResponse,
    SummaryRequest,
    SummaryResponse,
    UploadResponse,
)
from Code.core.session_manager import SessionStatus, session_manager
from Code.core.settings import ensure_runtime_dirs, session_upload_dir
from Code.rag.rag_pipeline import RAGPipeline


router = APIRouter()


def _get_ready_pipeline(session_id: str) -> RAGPipeline:
    try:
        state = session_manager.require_ready(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return state.pipeline


def _index_session_background(request: InitRequest) -> None:
    """
    Background indexing task.

    For MVP, this keeps the RAGPipeline object in memory.
    On a single EC2 backend process, this is acceptable.
    """
    session_id = request.session_id

    try:
        state = session_manager.get(session_id)
        session_manager.set_indexing(session_id)

        upload_dir = session_upload_dir(session_id)

        rag = RAGPipeline(top_k=12, load_llm=True)

        result = rag.build_index_from_path(
            input_path=upload_dir,
            use_ocr_for_all_pdfs=request.use_ocr_for_all_pdfs,
            include_video_frame_ocr=request.include_video_frame_ocr,
            video_frame_interval_sec=request.video_frame_interval_sec,
        )

        session_manager.set_ready(
            session_id=session_id,
            pipeline=rag,
            indexed_files=result.files,
        )

    except Exception as exc:
        session_manager.set_error(session_id, str(exc))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)) -> UploadResponse:
    ensure_runtime_dirs()

    state = session_manager.create_session()
    upload_dir = session_upload_dir(state.session_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    for uploaded in files:
        if not uploaded.filename:
            continue

        safe_name = Path(uploaded.filename).name
        destination = upload_dir / safe_name

        with destination.open("wb") as f:
            shutil.copyfileobj(uploaded.file, f)

        saved_files.append(destination)

    if not saved_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_manager.set_uploaded_files(state.session_id, saved_files)

    return UploadResponse(
        session_id=state.session_id,
        uploaded_files=[str(path) for path in saved_files],
    )


@router.post("/init", response_model=StatusResponse)
def initialize_session(
    request: InitRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse:
    try:
        state = session_manager.get(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.uploaded_files:
        raise HTTPException(status_code=400, detail="No uploaded files for this session")

    if state.status == SessionStatus.INDEXING:
        return StatusResponse(**state.to_dict())

    if state.status == SessionStatus.READY:
        return StatusResponse(**state.to_dict())

    session_manager.set_indexing(request.session_id)
    background_tasks.add_task(_index_session_background, request)

    state = session_manager.get(request.session_id)
    return StatusResponse(**state.to_dict())


@router.get("/status/{session_id}", response_model=StatusResponse)
def get_status(session_id: str) -> StatusResponse:
    try:
        state = session_manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return StatusResponse(**state.to_dict())


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    rag = _get_ready_pipeline(request.session_id)

    if request.source_contains or request.modality:
        answer = rag.answer_filtered(
            question=request.question,
            source_contains=request.source_contains,
            modality=request.modality,
            top_k=request.top_k,
        )
    else:
        answer = rag.answer(
            question=request.question,
            top_k=request.top_k,
        )

    return AskResponse(
        session_id=request.session_id,
        answer=answer,
    )


@router.post("/summary", response_model=SummaryResponse)
def summary(request: SummaryRequest) -> SummaryResponse:
    rag = _get_ready_pipeline(request.session_id)

    return SummaryResponse(
        session_id=request.session_id,
        summary=rag.summarize(),
    )


@router.post("/quiz", response_model=QuizResponse)
def quiz(request: QuizRequest) -> QuizResponse:
    rag = _get_ready_pipeline(request.session_id)

    return QuizResponse(
        session_id=request.session_id,
        quiz=rag.generate_quiz(
            num_questions=request.num_questions,
            difficulty=request.difficulty,
        ),
    )


@router.post("/flashcards", response_model=FlashcardResponse)
def flashcards(request: FlashcardRequest) -> FlashcardResponse:
    rag = _get_ready_pipeline(request.session_id)

    return FlashcardResponse(
        session_id=request.session_id,
        flashcards=rag.generate_flashcards(
            num_cards=request.num_cards,
        ),
    )


@router.get("/files/{session_id}", response_model=FilesResponse)
def files(session_id: str) -> FilesResponse:
    rag = _get_ready_pipeline(session_id)

    return FilesResponse(
        session_id=session_id,
        files_markdown=rag.list_files(),
    )


@router.get("/sources/{session_id}", response_model=SourcesResponse)
def sources(session_id: str) -> SourcesResponse:
    rag = _get_ready_pipeline(session_id)

    return SourcesResponse(
        session_id=session_id,
        sources_markdown=rag.list_sources(),
    )


@router.get("/quiz-prep/{session_id}", response_model=SummaryResponse)
def quiz_prep(session_id: str) -> SummaryResponse:
    rag = _get_ready_pipeline(session_id)

    return SummaryResponse(
        session_id=session_id,
        summary=rag.quiz_prep_by_source(),
    )