from __future__ import annotations

import re
from collections import defaultdict

from Code.core.schema import RAGChunk
from Code.core.settings import FLASHCARD_MAX_NEW_TOKENS, QUIZ_MAX_NEW_TOKENS
from Code.rag.orchestrator import QwenOrchestrator


def group_chunks_by_source(chunks: list[RAGChunk]) -> dict[str, list[RAGChunk]]:
    grouped: dict[str, list[RAGChunk]] = defaultdict(list)

    for chunk in chunks:
        source = chunk.source or "unknown"
        grouped[source].append(chunk)

    return dict(grouped)


def chunks_to_context(
    chunks: list[RAGChunk],
    max_chunks: int = 8,
) -> str:
    selected = chunks[:max_chunks]

    return "\n\n".join(
        f"[SOURCE: {chunk.source} | CHUNK: {chunk.chunk_id} | "
        f"MODALITY: {chunk.modality} | PAGE: {chunk.page} | "
        f"START: {chunk.start} | END: {chunk.end}]\n{chunk.text}"
        for chunk in selected
    )


def trim_numbered_questions(text: str, num_questions: int) -> str:
    """
    Best-effort post-processing to keep exactly N numbered questions.

    Works for outputs like:
    1. Question...
    2. Question...
    """
    pattern = r"(?m)^\s*(\d+)[\.\)]\s+"
    matches = list(re.finditer(pattern, text))

    if len(matches) <= num_questions:
        return text.strip()

    cutoff = matches[num_questions].start()
    return text[:cutoff].strip()


class QuizTools:
    """
    Higher-level study tools built on top of indexed chunks.

    This is where notebook-derived quiz/exam behavior belongs.
    """

    def __init__(self, orchestrator: QwenOrchestrator):
        self.orchestrator = orchestrator

    def summarize_each_source(
        self,
        chunks: list[RAGChunk],
        max_chunks_per_source: int = 8,
    ) -> str:
        if not chunks:
            return "No chunks indexed."

        grouped = group_chunks_by_source(chunks)
        outputs = []

        for source, source_chunks in sorted(grouped.items()):
            context = chunks_to_context(
                source_chunks,
                max_chunks=max_chunks_per_source,
            )

            summary = self.orchestrator.source_quiz_prep(
                source=source,
                context=context,
            )

            outputs.append(summary)

        return "\n\n".join(outputs)

    def generate_quiz_from_chunks(
        self,
        chunks: list[RAGChunk],
        num_questions: int = 10,
        difficulty: str = "mixed",
        max_chunks_per_source: int = 8,
    ) -> str:
        if not chunks:
            return "No chunks indexed."

        grouped = group_chunks_by_source(chunks)

        source_blocks = []

        for source, source_chunks in sorted(grouped.items()):
            context = chunks_to_context(
                source_chunks,
                max_chunks=max_chunks_per_source,
            )

            source_blocks.append(
                f"### SOURCE: {source}\n\n{context}"
            )

        full_context = "\n\n".join(source_blocks)

        system_prompt = (
            "You are an academic quiz generator. "
            "Use only the provided indexed coursework context. "
            "Create questions that test important concepts, not random details. "
            "Include the correct answer after every question. "
            "Include source, page, or timestamp when available."
        )

        user_prompt = f"""
Create a quiz from the indexed coursework.

Requirements:
- Create exactly {num_questions} questions.
- Difficulty: {difficulty}
- Mix multiple-choice and short-answer questions.
- Include the correct answer immediately after each question.
- Prioritize important concepts for quiz preparation.
- Include source, page, or timestamp when available.
- Do not ask questions from unsupported or missing context.

Context:
{full_context}

Return format:

## Quiz

1. **Question:** ...
   **Type:** MCQ / Short answer
   **Answer:** ...
   **Source reference:** ...

2. **Question:** ...
   **Type:** MCQ / Short answer
   **Answer:** ...
   **Source reference:** ...
"""

        quiz = self.orchestrator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=QUIZ_MAX_NEW_TOKENS,
        )

        return trim_numbered_questions(quiz, num_questions)

    def generate_exam_questions(
        self,
        chunks: list[RAGChunk],
        num_questions: int = 6,
        difficulty: str = "mixed",
        max_chunks_per_source: int = 10,
    ) -> str:
        if not chunks:
            return "No chunks indexed."

        grouped = group_chunks_by_source(chunks)

        source_blocks = []

        for source, source_chunks in sorted(grouped.items()):
            context = chunks_to_context(
                source_chunks,
                max_chunks=max_chunks_per_source,
            )

            source_blocks.append(
                f"### SOURCE: {source}\n\n{context}"
            )

        full_context = "\n\n".join(source_blocks)

        system_prompt = (
            "You are an exam question writer for an NLP coursework class. "
            "Use only the provided context. "
            "Write questions that test conceptual understanding and application."
        )

        user_prompt = f"""
Create exactly {num_questions} exam-style questions.

Difficulty: {difficulty}

Requirements:
- Prefer conceptual and application questions.
- Include the correct answer after every question.
- Include source, page, or timestamp when available.
- Use only the provided context.

Context:
{full_context}
"""

        questions = self.orchestrator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=QUIZ_MAX_NEW_TOKENS,
        )

        return trim_numbered_questions(questions, num_questions)

    def generate_flashcards_from_chunks(
        self,
        chunks: list[RAGChunk],
        num_cards: int = 12,
        max_chunks_per_source: int = 8,
    ) -> str:
        if not chunks:
            return "No chunks indexed."

        grouped = group_chunks_by_source(chunks)

        source_blocks = []

        for source, source_chunks in sorted(grouped.items()):
            context = chunks_to_context(
                source_chunks,
                max_chunks=max_chunks_per_source,
            )

            source_blocks.append(
                f"### SOURCE: {source}\n\n{context}"
            )

        full_context = "\n\n".join(source_blocks)

        system_prompt = (
            "You are an academic flashcard generator. "
            "Use only the provided coursework context. "
            "Create concise, useful flashcards."
        )

        user_prompt = f"""
Create exactly {num_cards} flashcards.

Requirements:
- Use front/back format.
- Prioritize definitions, formulas, algorithms, processes, and core concepts.
- Include source reference when useful.
- Use only the provided context.

Context:
{full_context}

Return format:

1. Front: ...
   Back: ...

2. Front: ...
   Back: ...
"""

        return self.orchestrator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=FLASHCARD_MAX_NEW_TOKENS,
        )