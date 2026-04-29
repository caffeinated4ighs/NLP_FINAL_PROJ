from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

CONFIG_DIR = PROJECT_ROOT / "Code" / "configs"
PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"


# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

USE_4BIT = True


# ---------------------------------------------------------------------
# RAG config
# ---------------------------------------------------------------------

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 12


# ---------------------------------------------------------------------
# Video/OCR config
# ---------------------------------------------------------------------

OCR_LANG = "en"

WHISPER_MODEL_NAME = "base"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

VIDEO_FRAME_INTERVAL_SEC = 10
PDF_RENDER_DPI = 200


# ---------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------

ANSWER_MAX_NEW_TOKENS = 300
SUMMARY_MAX_NEW_TOKENS = 800
QUIZ_MAX_NEW_TOKENS = 1400
FLASHCARD_MAX_NEW_TOKENS = 900


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_runtime_dirs() -> None:
    """
    Create runtime directories if they do not exist.
    These folders should exist locally but should not store committed data.
    """
    for path in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, INDEXES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def session_upload_dir(session_id: str) -> Path:
    return UPLOADS_DIR / session_id


def session_processed_dir(session_id: str) -> Path:
    return PROCESSED_DIR / session_id


def session_index_dir(session_id: str) -> Path:
    return INDEXES_DIR / session_id