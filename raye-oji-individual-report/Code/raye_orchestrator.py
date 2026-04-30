from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from Code.core.schema import RetrievalResult
from Code.core.settings import (
    ANSWER_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    PROMPTS_PATH,
    QUIZ_MAX_NEW_TOKENS,
    SUMMARY_MAX_NEW_TOKENS,
    USE_4BIT,
)


class QwenOrchestrator:
    """
    Qwen wrapper for answer/summary/quiz generation.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        use_4bit: bool = USE_4BIT,
        prompt_path: str | Path = PROMPTS_PATH,
    ) -> None:
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.prompts = self._load_prompts(prompt_path)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading LLM: {model_name}")
        self.model = self._load_model(model_name, use_4bit)
        self.model.eval()

    def _load_prompts(self, prompt_path: str | Path) -> dict:
        path = Path(prompt_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        required = ["answer", "summary", "quiz", "flashcards", "source_quiz_prep"]

        for key in required:
            if key not in prompts:
                raise ValueError(f"Missing prompt section: {key}")

            if "system" not in prompts[key] or "user_template" not in prompts[key]:
                raise ValueError(f"Prompt section {key} must have system and user_template.")

        return prompts

    def _load_model(self, model_name: str, use_4bit: bool):
        if use_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quant_config,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
        )

    def format_context(self, retrieved: list[RetrievalResult]) -> str:
        parts = []

        for i, result in enumerate(retrieved, start=1):
            chunk = result.chunk

            fields = []

            if chunk.source:
                fields.append(f"SOURCE: {chunk.source}")

            if chunk.page is not None:
                fields.append(f"PAGE: {chunk.page}")

            if chunk.start is not None:
                fields.append(f"START: {chunk.start:.2f}s")

            if chunk.end is not None:
                fields.append(f"END: {chunk.end:.2f}s")

            fields.append(f"MODALITY: {chunk.modality}")
            fields.append(f"BLOCK: {chunk.block_type}")
            fields.append(f"SCORE: {result.score:.4f}")

            header = " | ".join(fields)

            parts.append(
                f"[CONTEXT {i} | {header}]\n"
                f"{chunk.text}"
            )

        return "\n\n".join(parts)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = ANSWER_MAX_NEW_TOKENS,
        benchmark: bool = True,
    ) -> str:
        start_total = time.perf_counter()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        start_tokenize = time.perf_counter()
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)
        end_tokenize = time.perf_counter()

        input_tokens = inputs.input_ids.shape[1]

        start_generate = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        end_generate = time.perf_counter()

        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        output_tokens = len(generated_ids)

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        end_total = time.perf_counter()

        if benchmark:
            gen_time = end_generate - start_generate
            total_time = end_total - start_total
            tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0

            print("\n--- BENCHMARK ---")
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Tokenization time: {end_tokenize - start_tokenize:.2f}s")
            print(f"Generation time: {gen_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print(f"Tokens/sec: {tokens_per_sec:.2f}")

        return response

    def answer(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = self.format_context(retrieved)
        prompt = self.prompts["answer"]

        user_prompt = prompt["user_template"].format(
            question=question,
            context=context,
        )

        return self.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            max_new_tokens=ANSWER_MAX_NEW_TOKENS,
        )

    def summarize(self, retrieved: list[RetrievalResult]) -> str:
        context = self.format_context(retrieved)
        prompt = self.prompts["summary"]

        user_prompt = prompt["user_template"].format(
            context=context,
        )

        return self.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
        )

    def quiz(
        self,
        retrieved: list[RetrievalResult],
        num_questions: int = 8,
        difficulty: str = "mixed",
    ) -> str:
        context = self.format_context(retrieved)
        prompt = self.prompts["quiz"]

        user_prompt = prompt["user_template"].format(
            context=context,
            num_questions=num_questions,
            difficulty=difficulty,
        )

        return self.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            max_new_tokens=QUIZ_MAX_NEW_TOKENS,
        )

    def flashcards(
        self,
        retrieved: list[RetrievalResult],
        num_cards: int = 10,
    ) -> str:
        context = self.format_context(retrieved)
        prompt = self.prompts["flashcards"]

        user_prompt = prompt["user_template"].format(
            context=context,
            num_cards=num_cards,
        )

        return self.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            max_new_tokens=900,
        )

    def source_quiz_prep(self, source: str, context: str) -> str:
        prompt = self.prompts["source_quiz_prep"]

        user_prompt = prompt["user_template"].format(
            source=source,
            context=context,
        )

        return self.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            max_new_tokens=500,
        )