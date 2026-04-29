from __future__ import annotations

from collections import defaultdict

from Code.core.schema import IndexedFile, RAGChunk, RetrievalResult
from Code.core.settings import TOP_K
from Code.rag.embedder import BGEEmbedder
from Code.rag.vector_store import FAISSVectorStore


class Retriever:
    """
    Metadata-aware retrieval layer.

    Responsibilities:
    - build vector index from chunks
    - retrieve top-k chunks for normal RAG
    - filter retrieval by source/modality
    - expose deterministic source/file summaries
    """

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        vector_store: FAISSVectorStore | None = None,
        top_k: int = TOP_K,
    ) -> None:
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or FAISSVectorStore()
        self.top_k = top_k
        self.files: list[IndexedFile] = []

    @property
    def chunks(self) -> list[RAGChunk]:
        return self.vector_store.chunks

    def build(
        self,
        chunks: list[RAGChunk],
        files: list[IndexedFile] | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks provided for retrieval index.")

        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode_texts(texts)

        self.vector_store.build(chunks, embeddings)
        self.files = files or []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        query_embedding = self.embedder.encode_query(query)
        return self.vector_store.search(query_embedding, top_k=top_k or self.top_k)

    def retrieve_filtered(
        self,
        query: str,
        top_k: int | None = None,
        source_contains: str | None = None,
        modality: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve a large candidate set, then apply metadata filters.

        FAISS itself does not support metadata filtering, so this is the simplest
        reliable local approach for the MVP.
        """
        desired_k = top_k or self.top_k
        candidate_k = min(max(desired_k * 5, 50), max(len(self.chunks), desired_k))

        candidates = self.retrieve(query, top_k=candidate_k)

        filtered: list[RetrievalResult] = []

        for result in candidates:
            chunk = result.chunk

            if source_contains:
                source_text = f"{chunk.source or ''} {chunk.source_name or ''}".lower()
                if source_contains.lower() not in source_text:
                    continue

            if modality:
                if chunk.modality != modality:
                    continue

            filtered.append(result)

        return filtered[:desired_k]

    def list_files_table(self) -> str:
        if not self.files:
            return "No files have been indexed."

        lines = [
            "| # | File | Type | Chunks | Path |",
            "|---:|---|---|---:|---|",
        ]

        for i, item in enumerate(self.files, start=1):
            lines.append(
                f"| {i} | {item.name} | {item.file_type} | {item.chunk_count} | {item.path} |"
            )

        return "\n".join(lines)

    def source_summary_table(self) -> str:
        if not self.chunks:
            return "No chunks indexed."

        sources: dict[str, dict] = {}

        for chunk in self.chunks:
            source = chunk.source or "unknown"

            if source not in sources:
                sources[source] = {
                    "source_name": chunk.source_name or "unknown",
                    "chunks": 0,
                    "modalities": set(),
                    "pages": set(),
                    "timed_chunks": 0,
                }

            sources[source]["chunks"] += 1
            sources[source]["modalities"].add(chunk.modality)

            if chunk.page is not None:
                sources[source]["pages"].add(chunk.page)

            if chunk.start is not None or chunk.end is not None:
                sources[source]["timed_chunks"] += 1

        lines = [
            "| Source | Chunks | Modalities | Pages | Timed chunks |",
            "|---|---:|---|---:|---:|",
        ]

        for source, info in sorted(sources.items()):
            modalities = ", ".join(sorted(info["modalities"]))
            page_count = len(info["pages"])

            lines.append(
                f"| {info['source_name']} | {info['chunks']} | {modalities} | "
                f"{page_count} | {info['timed_chunks']} |"
            )

        return "\n".join(lines)

    def chunks_by_source(self) -> dict[str, list[RAGChunk]]:
        grouped: dict[str, list[RAGChunk]] = defaultdict(list)

        for chunk in self.chunks:
            source = chunk.source or "unknown"
            grouped[source].append(chunk)

        return dict(grouped)

    def save(self, index_dir: str) -> None:
        self.vector_store.save(index_dir)

    def load(self, index_dir: str) -> None:
        self.vector_store.load(index_dir)