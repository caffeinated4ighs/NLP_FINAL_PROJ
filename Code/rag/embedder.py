from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from Code.core.settings import EMBEDDING_MODEL_NAME


class BGEEmbedder:
    """
    Thin wrapper around BAAI/bge-m3.

    Keeps embedding separate from vector storage so we can swap FAISS/Chroma later.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        return np.asarray(embeddings, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embedding, dtype="float32")