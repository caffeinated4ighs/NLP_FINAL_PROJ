import time
from pathlib import Path

import torch
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from Code.scripts.schema import RetrievalResult


class QwenOrchestrator:
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_4bit: bool = True,
        prompt_config_path: str = "Code/configs/prompts.yaml",
    ):
        self.llm_model_name = llm_model_name
        self.prompts = self.load_prompts(prompt_config_path)

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
                torch_dtype=torch.float16,
                quantization_config=quant_config,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="auto",
                torch_dtype=torch.float32,
            )

        self.llm.eval()

    def load_prompts(self, prompt_config_path: str) -> dict:
        path = Path(prompt_config_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        required_sections = ["answer", "summary", "quiz"]

        for section in required_sections:
            if section not in prompts:
                raise ValueError(f"Missing prompt section: {section}")

            if "system" not in prompts[section]:
                raise ValueError(f"Missing system prompt for section: {section}")

            if "user_template" not in prompts[section]:
                raise ValueError(f"Missing user_template for section: {section}")

        return prompts

    def format_context(self, retrieved: list[RetrievalResult]) -> str:
        parts = []

        for i, result in enumerate(retrieved, start=1):
            chunk = result.chunk

            location_parts = []

            if chunk.source:
                location_parts.append(f"SOURCE: {chunk.source}")

            if chunk.page is not None:
                location_parts.append(f"PAGE: {chunk.page}")

            if chunk.start is not None:
                location_parts.append(f"START: {chunk.start:.2f}s")

            if chunk.end is not None:
                location_parts.append(f"END: {chunk.end:.2f}s")

            location_parts.append(f"MODALITY: {chunk.modality}")
            location_parts.append(f"BLOCK: {chunk.block_type}")
            location_parts.append(f"SCORE: {result.score:.4f}")

            header = " | ".join(location_parts)

            parts.append(
                f"[CONTEXT {i} | {header}]\n"
                f"{chunk.text}"
            )

        return "\n\n".join(parts)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 250,
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
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.llm.device)
        end_tokenize = time.perf_counter()

        input_tokens = inputs.input_ids.shape[1]

        start_generate = time.perf_counter()

        with torch.no_grad():
            output_ids = self.llm.generate(
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

        system_prompt = prompt["system"]
        user_prompt = prompt["user_template"].format(
            question=question,
            context=context,
        )

        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=250,
            benchmark=True,
        )

    def summarize(self, retrieved: list[RetrievalResult]) -> str:
        context = self.format_context(retrieved)

        prompt = self.prompts["summary"]

        system_prompt = prompt["system"]
        user_prompt = prompt["user_template"].format(
            context=context,
        )

        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=700,
            benchmark=True,
        )

    def generate_quiz(self, retrieved: list[RetrievalResult]) -> str:
        context = self.format_context(retrieved)

        prompt = self.prompts["quiz"]

        system_prompt = prompt["system"]
        user_prompt = prompt["user_template"].format(
            context=context,
        )

        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=900,
            benchmark=True,
        )