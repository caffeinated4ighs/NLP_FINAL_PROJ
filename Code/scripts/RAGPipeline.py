import torch
import faiss
import numpy as np
import fitz  # PyMuPDF

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name="BAAI/bge-m3",
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # safer for EC2 testing
        chunk_size=700,
        chunk_overlap=120,
        top_k=5,
        use_4bit=True,
    ):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.chunks = []
        self.index = None

        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)

        print(f"Loading tokenizer: {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        print(f"Loading LLM: {llm_model_name}")

        if use_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="auto",
                dtype=torch.float16,
                quantization_config=quant_config,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="auto",
                dtype=torch.float32,
            )

        self.llm.eval()

    def load_pdf(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"\n[PAGE {page_num}]\n{text}")

        return "\n".join(text_parts)

    def chunk_text(self, text):
        words = text.split()
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def build_index(self, pdf_path):
        print(f"Loading PDF: {pdf_path}")
        text = self.load_pdf(pdf_path)

        print("Chunking text...")
        self.chunks = self.chunk_text(text)

        if not self.chunks:
            raise ValueError("No chunks created from PDF.")

        print(f"Created {len(self.chunks)} chunks.")

        texts = [chunk["text"] for chunk in self.chunks]

        print("Creating embeddings...")
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.array(embeddings).astype("float32")

        print("Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print("Index ready.")

    def retrieve(self, query):
        if self.index is None:
            raise ValueError("Index is empty. Run build_index(pdf_path) first.")

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, self.top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def format_context(self, retrieved_chunks):
        parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            parts.append(
                f"[CONTEXT {i} | SCORE: {chunk['score']:.4f}]\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(parts)

    def generate(self, system_prompt, user_prompt, max_new_tokens=500):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            output_ids = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
            )

        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return response.strip()

    def answer(self, question):
        retrieved_chunks = self.retrieve(question)
        context = self.format_context(retrieved_chunks)

        system_prompt = (
            "You are a coursework-grounded RAG assistant. "
            "Answer only using the provided context. "
            "If the answer is not in the context, say you could not find it."
        )

        user_prompt = f"""
Question:
{question}

Context:
{context}

Answer clearly and concisely.
"""

        return self.generate(system_prompt, user_prompt)

    def summarize(self):
        retrieved_chunks = self.retrieve("Summarize the main topics in this coursework.")
        context = self.format_context(retrieved_chunks)

        system_prompt = (
            "You are an academic study assistant. "
            "Summarize only using the provided context."
        )

        user_prompt = f"""
Create a coursework summary using this context.

Include:
1. Main topics
2. Key definitions
3. Important concepts
4. What to review first

Context:
{context}
"""

        return self.generate(system_prompt, user_prompt, max_new_tokens=700)

    def generate_quiz(self):
        retrieved_chunks = self.retrieve("Generate a quiz from this coursework.")
        context = self.format_context(retrieved_chunks)

        system_prompt = (
            "You are an academic quiz generator. "
            "Use only the provided coursework context."
        )

        user_prompt = f"""
Create a quiz from this context.

Requirements:
- 8 questions
- Mix MCQ and short-answer
- Include answer key
- Mark each question easy, medium, or hard

Context:
{context}
"""

        return self.generate(system_prompt, user_prompt, max_new_tokens=900)