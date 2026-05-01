import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from Code.scripts.schema import RAGChunk, RetrievalResult


class BGERAGEmbedder:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        top_k: int = 5,
    ):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.model = SentenceTransformer(embedding_model_name)
        self.chunks: list[RAGChunk] = []
        self.index = None

    def chunk_text(
        self,
        text: str,
        source: str | None = None,
        modality: str = "pdf_text",
        page: int | None = None,
        block_type: str = "text",
    ) -> list[RAGChunk]:
        words = text.split()
        chunks: list[RAGChunk] = []

        start = 0
        chunk_id = 0

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(
                    RAGChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        modality=modality,
                        page=page,
                        block_type=block_type,
                    )
                )

            chunk_id += 1
            start += step

        return chunks

    def build_index_from_chunks(self, chunks: list[RAGChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if self.index is None:
            raise ValueError("Index is empty. Build the index first.")

        k = top_k or self.top_k

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(query_embedding, k)

        results: list[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                )
            )

        return results