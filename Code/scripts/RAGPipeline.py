import fitz
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class RAGPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer("BAAI/bge-m3")

        model_name = "Qwen/Qwen2.5-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )

        self.chunks = []
        self.index = None

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])

    def chunk_text(self, text, chunk_size=700, overlap=150):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    def build_index(self, pdf_path):
        text = self.extract_text(pdf_path)
        self.chunks = self.chunk_text(text)

        embeddings = self.embedding_model.encode(
            self.chunks,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query, k=3):
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")

        scores, idx = self.index.search(q_emb, k)

        return "\n\n".join([self.chunks[i] for i in idx[0]])

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer(self, question):
        context = self.retrieve(question)

        prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{question}
"""

        return self.generate(prompt)