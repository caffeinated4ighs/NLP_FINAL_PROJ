# Individual Contribution Report: Python RAG Pipeline and API

**Project:** NLP_FINAL_PROJ  
**Repository:** `caffeinated4ighs/NLP_FINAL_PROJ`  
**Branch context:** `master` as original prototype baseline; `dev` as current integrated version  
**Individual scope claimed:** Main Python RAG pipeline, indexing/init workflow, retrieval logic, and FastAPI API layer. Embedder/orchestrator model experimentation was shared with Raye, since both of us tested different model versions for performance and viability.  
**Scope not claimed:** Frontend implementation, UI styling, navbar/pages, general data-management ownership beyond runtime directories/files required by the Python pipeline, and the video/OCR ingestion pipeline owned by Aditi.

---

## 1. Project Overview

This project is a multimodal coursework assistant built around Retrieval-Augmented Generation (RAG). The system allows a user to upload coursework material and then use that uploaded material to ask questions, generate summaries, create quizzes, and build flashcards.

The central idea is that coursework is not always clean text. It can include text-based PDFs, scanned PDFs, images, slides, and lecture videos. Therefore, the backend pipeline was designed to normalize multiple file types into a common searchable format, index that content, retrieve the most relevant chunks for a user query, and pass that context to a language model for answer generation.

My individual contribution focuses on the Python backend work: the RAG pipeline, main pipeline, retrieval stack, initialization flow, and the FastAPI API endpoints. Model/embedder/orchestrator exploration was shared with Raye, while Aditi owned the video and OCR pipeline that produced text outputs consumed by the embedder that expose the pipeline to the application.

---

## 2. Baseline Before the Current Version

The earlier `master` branch served as the first working proof-of-concept. It validated the basic RAG workflow:

```text
PDF input
→ text extraction
→ chunking
→ embeddings
→ FAISS vector search
→ Qwen model response
```

That version was important because it proved the core concept. However, it was closer to a research/prototype script than a productized backend. Most of the functionality lived in a single `RAGPipeline.py` file, and the pipeline mainly targeted text-based PDF processing.

The limitations of the earlier baseline were:

```text
- Mostly single-file pipeline structure
- Limited separation of responsibilities
- No formal API layer
- No session-based upload/indexing lifecycle
- No async initialization flow
- No clean request/response models
- Limited multimodal ingestion structure
- Smaller model configuration
- Lower default retrieval coverage
```

The current `dev` branch builds on that prototype and turns it into a modular backend application.

---

## 3. Current Backend Architecture

The current Python backend is organized around clear responsibilities:

```text
Code/
├── api/
│   ├── api_server.py
│   ├── models.py
│   └── routes.py
├── core/
│   ├── schema.py
│   ├── session_manager.py
│   └── settings.py
├── ingestion/
│   ├── router.py
│   ├── pdf_loader.py
│   ├── document_ocr.py
│   ├── image_ocr.py
│   └── video_ingest.py
└── rag/
    ├── rag_pipeline.py
    ├── embedder.py
    ├── vector_store.py
    ├── retriever.py
    ├── orchestrator.py
    └── quiz_tools.py
```

This modular structure is a major improvement over the initial prototype. Each module now has a clear technical responsibility:

```text
api/        → FastAPI server, route handlers, request/response schemas
core/       → settings, runtime paths, shared schemas, session lifecycle
ingestion/  → conversion of uploaded files into text chunks
rag/        → embeddings, FAISS indexing, retrieval, model orchestration, study tools
```

This matters because RAG systems are easier to debug and extend when ingestion, retrieval, generation, and API concerns are separated.

---

## 4. My Claimed Contribution Area

My claimed contribution is the backend pipeline and API layer, specifically:

```text
- Main RAG pipeline design and integration
- Python model stack selection and wiring, shared with Raye for model/version performance checks
- Embedding model integration, shared with Raye because both of us tested different embedding/model configurations
- Consuming OCR/video text outputs handed off by Aditi and routing them into the embedding/retrieval pipeline
- FAISS vector indexing and retrieval flow
- Query answering through retrieved context
- Summary, quiz, and flashcard backend generation hooks
- Session initialization and indexing lifecycle
- FastAPI route design for upload/init/status/ask/study endpoints
- Backend request/response model structure
```

I am not claiming ownership of the frontend pages, UI components, navbar, styling, or client-side routing. The frontend consumes the API, but my contribution is the Python backend that powers the actual RAG behavior.

---

## 5. Main Pipeline Design

The main pipeline is coordinated through `Code/rag/rag_pipeline.py`.

The pipeline acts as the backend coordinator. It does not try to implement every low-level operation directly. Instead, it wires together the ingestion system, retriever, vector store, model orchestrator, and study-generation tools.

The pipeline flow is:

```text
Uploaded files
→ ingestion router
→ normalized RAG chunks
→ embedding model
→ FAISS vector index
→ retriever
→ LLM orchestrator
→ answer / summary / quiz / flashcards
```

The reason for this structure is technical separation. OCR, video processing, embedding, retrieval, and generation are different concerns. Keeping them separate makes the system easier to maintain and prevents the main pipeline from becoming an unmanageable script.

---

## 6. Ingestion and Chunk Normalization

The backend accepts different kinds of coursework files, but downstream retrieval needs one consistent representation. The solution was to normalize all extracted content into chunk objects.

A chunk represents a searchable unit of coursework content. For video and OCR inputs, Aditi owned the pipeline that extracted usable text; my backend pipeline consumed that text and routed it into the embedder/retriever. A chunk can come from:

```text
- Text PDF pages
- OCR output from scanned PDFs
- OCR output from images
- Video transcripts
- OCR output from sampled video frames
```

The technical reason for this is that the retriever and embedding model should not need to know whether text came from a PDF, image, or video. Once content becomes a normalized chunk, the rest of the pipeline can treat it consistently.

This design avoids writing separate RAG systems for each modality.

Instead of:

```text
PDF QA system
Image QA system
Video QA system
```

the backend uses:

```text
All uploaded content
→ normalized chunks
→ one retrieval and generation pipeline
```

That is a better architecture for a coursework assistant because real course material is mixed-format.

---

## 7. Model Stack

**Ownership note:** The embedder and orchestrator work should be described as shared with Raye. We both worked with different versions of the model stack to check performance, feasibility, and output quality. Aditi owns the video and OCR pipeline; her components hand extracted text to the backend pipeline, and my claimed scope begins where that text is normalized, embedded, retrieved, and exposed through the API.

The current backend model stack is:

```text
Embedding model: BAAI/bge-m3
LLM:             Qwen/Qwen2.5-7B-Instruct
Vector store:    FAISS
Runtime:         PyTorch + CUDA
Quantization:    4-bit loading enabled
```

These choices were made for practical and technical reasons.

### 7.1 Embedding Model: BAAI/bge-m3

The embedding model converts both coursework chunks and user questions into vectors. This allows semantic retrieval instead of keyword-only search.

BGE-M3 was selected because it is a strong general-purpose embedding model and is appropriate for academic/coursework-style text. The retrieval model needs to handle definitions, lecture notes, textbook-like explanations, and technical terms. A semantic embedding model is better for this than simple keyword matching.

Using an embedding model also makes the system efficient:

```text
Indexing phase:
- Embed all chunks once

Question phase:
- Embed the user question once
- Search the vector index
- Send only relevant chunks to the LLM
```

This is much better than passing entire documents directly into the LLM.

### 7.2 LLM: Qwen2.5-7B-Instruct

The LLM is used after retrieval. It receives the retrieved context and generates the final answer, summary, quiz, or flashcards.

The current version uses Qwen2.5-7B-Instruct because the project requires more than short answers. It needs structured academic outputs such as:

```text
- Coursework summaries
- Concept explanations
- Quiz questions
- Flashcards
- Source-grounded answers
```

The earlier prototype used a smaller model, which was easier to run but weaker for these tasks. The 7B model gives better instruction following and output quality while remaining feasible on GPU with 4-bit loading.

### 7.3 4-bit Loading

The 7B model is heavier than the original prototype model. To make it practical on the EC2 GPU environment, 4-bit loading is enabled.

The reason is straightforward:

```text
Full precision:
- Higher VRAM usage
- More likely to run out of GPU memory

4-bit:
- Lower VRAM usage
- More practical for deployment
- Still good enough for the project MVP
```

This was a deployment-aware modeling decision, not just a model-quality decision.

---

## 8. Vector Store Choice: FAISS

The project uses FAISS as the vector search backend.

FAISS was chosen instead of ChromaDB because the current project needed a local, simple, and reliable vector search layer. FAISS works well for this MVP because:

```text
- It is lightweight
- It runs locally
- It does not require a separate database service
- It avoids extra server configuration
- It avoids AWS/IAM/database setup issues
- It is fast for similarity search
- It is easy to persist as index files
```

ChromaDB would be useful later if the project needed richer persistent metadata querying, multiple users, or a more database-like vector store. However, at the current stage, ChromaDB would add extra moving parts without being necessary for the final demo.

The current design keeps FAISS as a simple vector index and stores chunk metadata separately. That gives enough functionality for the project while keeping the backend easier to debug.

---

## 9. Retrieval Logic

The retriever is responsible for finding the most relevant chunks for a user question.

The retrieval process is:

```text
1. User asks a question
2. Question is embedded with the same embedding model
3. FAISS searches for nearest chunk vectors
4. Top matching chunks are returned
5. Retrieved chunks are inserted into the prompt
6. LLM generates a grounded answer
```

The retriever also supports optional filters such as source filtering or modality filtering. For example, a user can ask the backend to retrieve only from a specific file or only from a specific content type.

FAISS itself does not natively support the kind of metadata filtering used in this MVP. Therefore, the retriever searches a larger candidate pool first and then filters the results in Python. This is a reasonable MVP tradeoff because it keeps FAISS simple while still allowing useful source/modality filtering.

---

## 10. What `top_k` Does

`top_k` controls how many retrieved chunks are passed into the LLM for a given query.

For example:

```text
top_k = 5
```

means:

```text
Retrieve the 5 most relevant chunks
→ give those chunks to the LLM
→ generate the answer from that context
```

This parameter directly affects answer quality, speed, and context noise.

### If `top_k` is too low

The model may not receive enough evidence.

Possible problems:

```text
- Missing key definitions
- Incomplete summaries
- Weak quiz questions
- Answers based on only one small part of the coursework
- Higher chance of saying relevant content was not found
```

### If `top_k` is too high

The model may receive too much context.

Possible problems:

```text
- Slower responses
- More GPU memory pressure
- Noisier prompts
- Less focused answers
- Higher chance of mixing unrelated topics
```

### Why the current default is 5

The current default is:

```text
TOP_K = 5
```

This value was selected as a practical MVP default because the backend has to balance answer quality with latency and GPU memory constraints. Since the pipeline may already be working with OCR-derived text, transcript-derived text, embeddings, and a local LLM, retrieving too much context can make prompts slower and noisier.

A default of 5 keeps the prompt focused on the strongest matches while still giving the model enough evidence for normal question answering. For broader tasks such as full-course summaries or quiz generation, the API can still accept a larger `top_k` value when needed.

It is a conservative default:

```text
top_k = 5
→ faster answers
→ focused context
→ lower prompt noise
→ still adjustable for broader retrieval tasks
```

---

## 11. API Layer

The FastAPI layer exposes the pipeline to the application.

The main API responsibilities are:

```text
- Accept uploaded files
- Create a session
- Initialize indexing in the background
- Report session status
- Accept questions
- Return answers
- Generate summaries
- Generate quizzes
- Generate flashcards
- Return indexed files and source summaries
```

The main backend API files are:

```text
Code/api/api_server.py
Code/api/routes.py
Code/api/models.py
```

### 11.1 API Server

`api_server.py` creates the FastAPI application and mounts the API router under `/api`.

It also configures CORS so that the frontend can call the backend while running locally. This is necessary because the frontend and backend run on different ports during development.

### 11.2 API Models

`models.py` defines the request and response models using Pydantic.

This is important because the frontend and backend need a clear contract. Instead of passing unstructured dictionaries, each endpoint expects and returns typed objects.

Examples of API models include:

```text
UploadResponse
InitRequest
StatusResponse
AskRequest
AskResponse
SummaryRequest
SummaryResponse
QuizRequest
QuizResponse
FlashcardRequest
FlashcardResponse
```

### 11.3 API Routes

`routes.py` contains the backend route handlers.

The key endpoints are:

```text
GET  /api/health
POST /api/upload
POST /api/init
GET  /api/status/{session_id}
POST /api/ask
POST /api/summary
POST /api/quiz
POST /api/flashcards
GET  /api/files/{session_id}
GET  /api/sources/{session_id}
GET  /api/quiz-prep/{session_id}
```

The API is designed around a session lifecycle:

```text
upload
→ init
→ indexing
→ ready
→ ask / summary / quiz / flashcards
```

This is better than trying to answer immediately at upload time because model loading, OCR, video processing, chunking, and indexing can take time.

---

## 12. Initialization and Background Indexing

One of the most important backend design improvements was separating initialization from querying.

The API does not try to do all expensive work inside `/ask`. Instead:

```text
POST /api/upload
→ stores uploaded files and returns session_id

POST /api/init
→ starts background indexing

GET /api/status/{session_id}
→ reports whether indexing is done

POST /api/ask
→ only works once session is ready
```

This design is technically better because indexing can be slow. It may involve:

```text
- OCR
- video frame processing
- transcript extraction
- chunk generation
- embedding model calls
- FAISS index construction
- LLM loading
```

If all of that happened during a question request, the frontend would appear frozen and the API would be harder to debug.

By splitting initialization from querying, the user flow becomes more reliable:

```text
Heavy work happens once
→ session becomes ready
→ interactive features become available
```

This is the same architecture used in many practical RAG applications.

---

## 13. Session Management

The session manager tracks the state of each upload session.

A session can move through states like:

```text
created
→ uploaded
→ indexing
→ ready
```

or enter an error state if indexing fails.

This is necessary because the API needs to know:

```text
- Which files belong to the current session
- Whether indexing has started
- Whether the pipeline is ready
- Whether a previous error occurred
- Which RAGPipeline object should answer the request
```

For the MVP, the ready `RAGPipeline` object is stored in memory. This is acceptable for a single EC2 backend process and final-project demo. A future production version could persist session state and indexes more robustly.

---

## 14. Study Tools

The backend supports several study-oriented outputs:

```text
- Direct question answering
- Coursework summary
- Quiz generation
- Flashcard generation
- Quiz preparation by source
```

These are built on top of the same core pipeline.

The important technical point is that these are not separate systems. They reuse the same indexed coursework and retrieval context. Different endpoints call different pipeline methods, but the source material and model infrastructure are shared.

This makes the system cleaner:

```text
One index
One retriever
One model orchestration layer
Multiple study features
```

---

## 15. Why the Current Version Is Better Than the Prototype

The current version is better than the original `master` prototype because it turns the idea into a real backend application.

### Prototype version

```text
- Single-file style pipeline
- Mostly PDF-focused
- Local/manual usage
- Smaller model
- Conservative retrieval coverage
- No formal API lifecycle
```

### Current version

```text
- Modular Python package
- API-based workflow
- Session state
- Background indexing
- Multimodal ingestion structure
- Stronger 7B model
- More configurable retrieval through request-level `top_k`
- Summary/quiz/flashcard endpoints
- Frontend-ready backend contract
```

The key technical improvement is that the current version separates preparation from interaction:

```text
Initialization phase:
- ingest
- chunk
- embed
- index
- load model

Interaction phase:
- retrieve
- generate
- return answer
```

This directly improves usability and backend maintainability.

---

## 16. Why These Choices Were Rational

The final technical choices were made for practical reasons:

| Decision | Reason |
|---|---|
| FastAPI | Clean Python API layer, upload support, typed models, background tasks |
| Session-based workflow | Needed to track upload, indexing, ready, and error states |
| Async/background initialization | Prevents long indexing from blocking question answering |
| BAAI/bge-m3 | Strong semantic retrieval model for academic/coursework text |
| FAISS | Local, simple, fast vector search without database overhead |
| Qwen2.5-7B-Instruct | Better output quality for summaries, explanations, and quizzes |
| 4-bit loading | Makes 7B model feasible on limited GPU memory |
| `top_k = 5` | Conservative default that keeps retrieval focused and responses faster, while allowing larger values per request when needed |
| Modular Python structure | Easier debugging, testing, and future extension |

These are not arbitrary choices. Each one addresses a specific project constraint: limited time, EC2 deployment, GPU memory, frontend responsiveness, and the need to support real coursework files.

---

## 17. Current Limitations

The current backend is appropriate for an MVP, but it has limitations:

```text
- In-memory session pipeline is not durable across backend restarts
- FAISS metadata filtering is handled manually after retrieval
- Large video/OCR workloads can still be slow
- Multi-user production scaling would need persistent session/index storage
- Model quality depends on GPU availability and quantized inference stability
```

These limitations are acceptable for the project scope. The current system prioritizes a working, demonstrable, technically coherent RAG application.

---

## 18. Possible Future Improvements

Future backend improvements could include:

```text
- Persisting indexes per session and reloading them after restart
- Adding a durable database for session metadata
- Improving metadata-aware retrieval
- Adding reranking after FAISS search
- Supporting syllabus-weighted quiz generation more explicitly
- Adding streaming responses for long answers
- Adding queue-based background jobs for long OCR/video tasks
- Adding better evaluation metrics for retrieval and quiz quality
```

These would move the project from MVP toward production readiness.

---

## 19. Final Individual Contribution Summary

My contribution was the Python backend intelligence layer of the system. I focused on turning the initial RAG prototype into a structured backend pipeline that could support a real application.

The main work I am claiming is:

```text
- Designing and integrating the main RAG pipeline
- Connecting normalized text outputs from ingestion into embedding, retrieval, and model generation
- Setting up the retrieval stack with BGE-M3 and FAISS
- Sharing embedder/orchestrator model-version testing with Raye to evaluate performance and viability
- Consuming Aditi's OCR/video text outputs in the embedding and retrieval pipeline
- Implementing the query and study-generation backend flow
- Designing the initialization/indexing lifecycle
- Exposing the backend through FastAPI endpoints
- Supporting session-aware interaction between uploaded coursework and model outputs
```

The final result is a backend that can ingest coursework, build a semantic index, retrieve relevant context, and generate useful study outputs through API endpoints. This transformed the project from a local RAG prototype into a deployable multimodal coursework assistant backend.

---

## 20. Repository References

- `Code/rag/rag_pipeline.py` — main RAG pipeline coordinator
- `Code/rag/embedder.py` — embedding model wrapper
- `Code/rag/vector_store.py` — FAISS vector store
- `Code/rag/retriever.py` — retrieval and filtering logic
- `Code/rag/orchestrator.py` — LLM orchestration
- `Code/rag/quiz_tools.py` — quiz and study-generation utilities
- `Code/api/api_server.py` — FastAPI application setup
- `Code/api/routes.py` — API route handlers
- `Code/api/models.py` — Pydantic request/response models
- `Code/core/settings.py` — model, RAG, OCR, and runtime configuration
- `Code/core/session_manager.py` — session lifecycle and state management
- `Code/core/schema.py` — shared chunk and indexing schemas
