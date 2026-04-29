# Learn Loop — Multimodal Coursework RAG Tutor

Learn Loop is a multimodal Retrieval-Augmented Generation (RAG) project for uploading coursework material and turning it into searchable study support. The system accepts files such as text PDFs, scanned/image-based PDFs, images, and videos, indexes the extracted content, and exposes API/UI features for:

- coursework question answering
- summarization
- quiz generation
- flashcard generation
- source/file inspection

The application is split into a FastAPI backend and a Next.js frontend.

---

## Repository layout

```text
NLP_FINAL_PROJ/
├── Code/
│   ├── api/                 # FastAPI app, routes, request/response models
│   ├── cli/                 # Terminal interface for local ingestion and Q&A
│   ├── configs/             # Prompt/config files
│   ├── core/                # Shared schema, settings, and session state
│   ├── frontend/            # Next.js frontend
│   ├── ingestion/           # PDF, OCR, image, and video ingestion logic
│   ├── rag/                 # Embedding, FAISS retrieval, LLM orchestration, quiz tools
│   └── __init__.py
├── data/
│   ├── indexes/             # Runtime vector index outputs
│   ├── processed/           # Runtime processed files/chunks
│   └── uploads/             # Runtime uploaded files, grouped by session_id
├── .env                     # Local environment variables; do not commit secrets
├── .gitignore
├── README.md
├── requirements.txt         # Backend Python dependency pins
└── uv.lock                  # uv lockfile
```

---

## System architecture

```text
Frontend upload page
        ↓
POST /api/upload
        ↓
Backend creates a session_id and saves files under data/uploads/<session_id>/
        ↓
POST /api/init
        ↓
FastAPI background task starts indexing
        ↓
Code/ingestion/router.py dispatches files by type
        ↓
PDF text extraction / scanned PDF OCR / image OCR / video ingestion
        ↓
All extracted content becomes RAGChunk objects
        ↓
Code/rag/embedder.py creates embeddings
        ↓
Code/rag/vector_store.py builds/searches a FAISS index
        ↓
Code/core/session_manager.py marks the session as ready
        ↓
Frontend polls GET /api/status/{session_id}
        ↓
When ready, frontend enables:
    POST /api/ask
    POST /api/summary
    POST /api/quiz
    POST /api/flashcards
    GET  /api/files/{session_id}
    GET  /api/sources/{session_id}
```

The current MVP keeps the constructed `RAGPipeline` object in process memory per session. This is acceptable for a single-process EC2 demo, but a backend restart will lose active session pipelines unless index/session persistence is added and reloaded.

---

## Backend

### Main entrypoint

```text
Code/api/api_server.py
```

Creates the FastAPI app, enables CORS for local frontend development, mounts the API router under `/api`, and exposes a root health response.

Working backend command from the project root:

```bash
uv run uvicorn Code.api.api_server:app --host 0.0.0.0 --port 8000
```

For local-only development:

```bash
uv run uvicorn Code.api.api_server:app --reload --host 127.0.0.1 --port 8000
```

### API routes

```text
Code/api/routes.py
```

Defines the backend workflow routes.

| Route | Method | Purpose |
|---|---:|---|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload coursework files and create a session |
| `/api/init` | POST | Start background indexing for a session |
| `/api/status/{session_id}` | GET | Poll indexing/session status |
| `/api/ask` | POST | Ask a retrieval-backed question |
| `/api/summary` | POST | Generate a summary of indexed coursework |
| `/api/quiz` | POST | Generate quiz questions |
| `/api/flashcards` | POST | Generate flashcards |
| `/api/files/{session_id}` | GET | List indexed files |
| `/api/sources/{session_id}` | GET | Summarize indexed sources |
| `/api/quiz-prep/{session_id}` | GET | Generate quiz-prep notes by source |

### API models

```text
Code/api/models.py
```

Contains Pydantic request/response schemas for the API contract.

Important request models:

```json
{
  "session_id": "uuid",
  "question": "summarize the uploaded coursework",
  "top_k": 12,
  "source_contains": null,
  "modality": null
}
```

`top_k` must be between `1` and `20`.

For unfiltered Q&A, keep both filters as `null`:

```json
{
  "session_id": "5d1ea84c-53de-48d1-a7c8-f180c21c6bfc",
  "question": "summarize the uploaded coursework",
  "top_k": 12,
  "source_contains": null,
  "modality": null
}
```

Do **not** send placeholder strings like `"string"` for `source_contains` or `modality`, because the backend treats those as real filters. If no indexed chunk matches that filter, the backend returns:

```text
I could not find matching indexed content for that filter.
```

### Core backend files

#### `Code/core/settings.py`

Centralizes project paths, model names, chunking parameters, OCR/video settings, and generation limits.

Notable defaults:

- embedding model: `BAAI/bge-m3`
- LLM: `Qwen/Qwen2.5-7B-Instruct`
- chunk size: `700`
- chunk overlap: `120`
- default retrieval `top_k`: `12`
- video frame interval: `10` seconds

#### `Code/core/schema.py`

Defines shared internal data structures such as `RAGChunk` and indexed-file metadata. All modalities are normalized into chunk objects before retrieval.

#### `Code/core/session_manager.py`

Tracks each API session through statuses such as uploaded, indexing, ready, or error. It stores uploaded files, indexed files, and the active in-memory `RAGPipeline`.

---

## Ingestion layer

```text
Code/ingestion/
├── document_ocr.py
├── image_ocr.py
├── pdf_loader.py
├── router.py
└── video_ingest.py
```

### `pdf_loader.py`

Handles text-based PDFs using PyMuPDF. It extracts page text and turns it into chunks.

### `document_ocr.py`

Handles scanned or image-based PDFs through OCR.

### `image_ocr.py`

Extracts text from standalone image files such as screenshots, diagrams, or photographed notes.

### `video_ingest.py`

Handles video ingestion. It supports video-related extraction and optional sampled frame OCR based on the configured frame interval.

### `router.py`

Dispatches each uploaded file to the correct ingestion path based on file type and flags such as:

- `use_ocr_for_all_pdfs`
- `include_video_frame_ocr`
- `video_frame_interval_sec`

---

## RAG layer

```text
Code/rag/
├── embedder.py
├── orchestrator.py
├── quiz_tools.py
├── rag_pipeline.py
├── retriever.py
└── vector_store.py
```

### `rag_pipeline.py`

The main coordinator. It wires together:

```text
ingestion → chunks → embeddings → FAISS index → retrieval → LLM answer/study generation
```

Primary methods include:

- `build_index_from_path(...)`
- `answer(...)`
- `answer_filtered(...)`
- `summarize(...)`
- `generate_quiz(...)`
- `generate_flashcards(...)`
- `quiz_prep_by_source(...)`
- `list_files()`
- `list_sources()`

### `embedder.py`

Wraps the embedding model and converts text chunks or user queries into vector representations.

### `vector_store.py`

Owns the FAISS vector index and nearest-neighbor search.

### `retriever.py`

Provides retrieval over indexed chunks. It supports normal top-k retrieval and filtered retrieval by source substring and/or modality.

Filtering is applied after candidate retrieval. That means exact placeholder values like `"string"` will usually filter everything out.

### `orchestrator.py`

Handles LLM prompt construction and generation using Qwen. It receives retrieved context and produces final responses.

### `quiz_tools.py`

Contains study-generation helpers for quizzes, exam-style questions, flashcards, and source-level study summaries.

---

## CLI workflow

```text
Code/cli/ingest_data.py
```

Provides a terminal workflow for indexing files and asking questions without the frontend.

It supports commands such as:

- `files`
- `sources`
- `summary`
- `quiz prep`
- `quiz`
- `exam`
- `flashcards`

It also supports filtered question prefixes such as:

```text
pdf: explain this topic
ocrpdf: summarize the scanned content
image: what does this image say?
video: summarize the video
frame: what appears in the sampled frames?
source=filename.pdf:: summarize this source
```

---

## Frontend

The frontend is a Next.js app under:

```text
Code/frontend/
```

### Frontend tree

```text
Code/frontend/
├── app/
│   ├── flashcards/
│   │   ├── flashcards-page-client.tsx
│   │   └── page.tsx
│   ├── quiz/
│   │   ├── page.tsx
│   │   └── quiz-page-client.tsx
│   ├── status/
│   │   ├── page.tsx
│   │   └── status-page-client.tsx
│   ├── summary/
│   │   ├── page.tsx
│   │   └── summary-page-client.tsx
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── markdown-math.tsx
│   └── ui/
├── lib/
│   └── utils.ts
├── public/
├── package.json
├── next.config.ts
├── tsconfig.json
└── postcss.config.mjs
```

### Main frontend routes

| Route | File | Purpose |
|---|---|---|
| `/` | `app/page.tsx` | Upload/start page |
| `/status` | `app/status/page.tsx` + `status-page-client.tsx` | Poll backend status and ask questions when ready |
| `/summary` | `app/summary/page.tsx` + `summary-page-client.tsx` | Generate coursework summary |
| `/quiz` | `app/quiz/page.tsx` + `quiz-page-client.tsx` | Generate quiz |
| `/flashcards` | `app/flashcards/page.tsx` + `flashcards-page-client.tsx` | Generate flashcards |

### Math rendering

```text
Code/frontend/components/markdown-math.tsx
```

This component renders backend Markdown and math output using:

- `react-markdown`
- `remark-math`
- `rehype-katex`
- `katex`

This prevents equations from displaying as raw or distorted text.

---

## Setup

### 1. Clone and switch to dev branch

```bash
git clone https://github.com/caffeinated4ighs/NLP_FINAL_PROJ.git
cd NLP_FINAL_PROJ
git checkout dev
```

### 2. Backend setup

From the project root:

```bash
uv sync
```

If using `requirements.txt` instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the backend:

```bash
uv run uvicorn Code.api.api_server:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### 3. Frontend setup

```bash
cd Code/frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open:

```text
http://localhost:3000
```

If running on EC2, expose or tunnel port `3000` for the frontend and port `8000` for the backend.

---

## API usage examples

### Upload files

```bash
curl -X POST "http://127.0.0.1:8000/api/upload" \
  -F "files=@/path/to/coursework.pdf" \
  -F "files=@/path/to/lecture.mp4"
```

Response includes a `session_id`.

### Start indexing

```bash
curl -X POST "http://127.0.0.1:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "use_ocr_for_all_pdfs": false,
    "include_video_frame_ocr": true,
    "video_frame_interval_sec": 10
  }'
```

### Check status

```bash
curl "http://127.0.0.1:8000/api/status/YOUR_SESSION_ID"
```

Wait until:

```json
{
  "ready": true,
  "status": "ready"
}
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "question": "summarize the uploaded coursework",
    "top_k": 12,
    "source_contains": null,
    "modality": null
  }'
```

### Ask with a source filter

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "question": "summarize this source",
    "top_k": 12,
    "source_contains": "lecture",
    "modality": null
  }'
```

### Ask with a modality filter

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "question": "what was extracted from the image content?",
    "top_k": 12,
    "source_contains": null,
    "modality": "image"
  }'
```

Only use a modality value that actually exists in indexed chunks.

### Generate a summary

```bash
curl -X POST "http://127.0.0.1:8000/api/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID"
  }'
```

### Generate a quiz

```bash
curl -X POST "http://127.0.0.1:8000/api/quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "num_questions": 10,
    "difficulty": "mixed"
  }'
```

### Generate flashcards

```bash
curl -X POST "http://127.0.0.1:8000/api/flashcards" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "num_cards": 12
  }'
```

---

## Development notes

### Backend and frontend ports

Default local ports:

```text
Backend:  http://127.0.0.1:8000
Frontend: http://127.0.0.1:3000
```

On EC2, bind to `0.0.0.0`:

```bash
uv run uvicorn Code.api.api_server:app --host 0.0.0.0 --port 8000
npm run dev -- --host 0.0.0.0
```

Make sure AWS security groups allow inbound traffic for the ports you need, or use SSH tunneling.

### Next.js Node version

The frontend uses Next.js 16, which requires a modern Node.js version. Use Node `>=20.9.0`.

Example with `nvm`:

```bash
nvm install 20
nvm use 20
node -v
```

### Common issue: filtered ask returns no content

If `/api/ask` returns:

```text
I could not find matching indexed content for that filter.
```

Check the request body. This is usually caused by sending placeholder filters:

```json
{
  "source_contains": "string",
  "modality": "string"
}
```

Use this instead for normal questions:

```json
{
  "source_contains": null,
  "modality": null
}
```

### Common issue: session not ready

If feature endpoints return a conflict or the frontend is waiting, poll:

```bash
curl http://127.0.0.1:8000/api/status/YOUR_SESSION_ID
```

Do not call `/api/ask`, `/api/summary`, `/api/quiz`, or `/api/flashcards` until `ready` is `true`.

### Common issue: backend restart loses session

Current session pipelines are stored in memory. Restarting Uvicorn clears active pipelines. Re-upload and re-initialize, or implement persistent index loading through `RAGPipeline.save_index()` and `RAGPipeline.load_index()`.

---

## Recommended demo flow

1. Start backend.
2. Start frontend.
3. Upload a mix of coursework files:
   - text PDF
   - scanned/image PDF
   - image
   - video
4. Click/init indexing.
5. Wait on the status page until the session is ready.
6. Ask questions from the interactive query area.
7. Generate summary, quiz, and flashcards.
8. Use files/sources endpoints to inspect what was indexed.

---

## Current limitations

- Active RAG pipelines are stored in process memory.
- Session recovery after backend restart is not complete.
- Metadata filtering depends on exact indexed source/modality values.
- Large video and OCR-heavy inputs can take time to process.
- Running Qwen2.5-7B requires sufficient GPU memory and correct PyTorch/CUDA alignment.
- FAISS search itself is vector-based; metadata filtering happens after candidate retrieval.

---

## Suggested next improvements

- Persist FAISS indexes per session and reload them after restart.
- Add a persistent session database or lightweight JSON manifest.
- Add a `/api/modalities/{session_id}` endpoint so the frontend can show valid modality filters.
- Add frontend display of indexed source names to avoid bad `source_contains` filters.
- Add background-job progress percentages for long OCR/video indexing.
- Add production deployment through systemd + Nginx reverse proxy.
- Move API base URL into frontend environment configuration.
