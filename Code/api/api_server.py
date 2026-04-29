from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Code.api.routes import router


app = FastAPI(
    title="NLP Final Project Multimodal RAG API",
    version="0.1.0",
    description="Backend API for coursework RAG over PDFs, OCR images, and videos.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "NLP Final Project RAG API is running.",
    }