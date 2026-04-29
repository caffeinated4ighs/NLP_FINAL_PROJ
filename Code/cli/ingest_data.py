from __future__ import annotations

import argparse

from Code.core.settings import DATA_DIR
from Code.rag.rag_pipeline import RAGPipeline


def print_block(title: str, body: str) -> None:
    print(f"\n{title}")
    print(body)
    print("\n" + "-" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local CLI for multimodal coursework RAG."
    )

    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Folder or file to ingest. Defaults to data/.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of chunks to retrieve for normal questions.",
    )

    parser.add_argument(
        "--ocr-all-pdfs",
        action="store_true",
        help="Force OCR on all PDFs, including text-based PDFs.",
    )

    parser.add_argument(
        "--no-frame-ocr",
        action="store_true",
        help="Disable OCR on sampled video frames.",
    )

    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Seconds between sampled video frames.",
    )

    return parser.parse_args()


def route_question(rag: RAGPipeline, question: str) -> str:
    q_lower = question.lower().strip()

    if q_lower in {"files", "list files", "indexed files", "what files are indexed"}:
        return rag.list_files()

    if q_lower in {"sources", "source summary", "context sources"}:
        return rag.list_sources()

    if q_lower in {"summary", "summarize", "summarize all"}:
        return rag.summarize()

    if q_lower in {"quiz prep", "summarize each file", "break down each file"}:
        return rag.quiz_prep_by_source()

    if q_lower in {"quiz", "make quiz", "generate quiz", "sample quiz"}:
        return rag.generate_quiz(num_questions=10, difficulty="mixed")

    if q_lower in {"exam", "exam questions", "generate exam questions"}:
        return rag.generate_exam_questions(num_questions=6, difficulty="mixed")

    if q_lower in {"flashcards", "make flashcards", "generate flashcards"}:
        return rag.generate_flashcards(num_cards=12)

    if q_lower.startswith("pdf:"):
        actual_question = question.split(":", 1)[1].strip()
        return rag.answer_filtered(
            actual_question,
            modality="pdf_text",
        )

    if q_lower.startswith("ocrpdf:"):
        actual_question = question.split(":", 1)[1].strip()
        return rag.answer_filtered(
            actual_question,
            modality="pdf_ocr",
        )

    if q_lower.startswith("video:"):
        actual_question = question.split(":", 1)[1].strip()
        return rag.answer_filtered(
            actual_question,
            modality="video_transcript",
        )

    if q_lower.startswith("frame:"):
        actual_question = question.split(":", 1)[1].strip()
        return rag.answer_filtered(
            actual_question,
            modality="video_frame_ocr",
        )

    if q_lower.startswith("image:"):
        actual_question = question.split(":", 1)[1].strip()
        return rag.answer_filtered(
            actual_question,
            modality="image_ocr",
        )

    if q_lower.startswith("source=") and "::" in question:
        source_filter, actual_question = question.split("::", 1)
        source_filter = source_filter.replace("source=", "").strip()
        actual_question = actual_question.strip()

        return rag.answer_filtered(
            actual_question,
            source_contains=source_filter,
        )

    return rag.answer(question)


def main() -> None:
    args = parse_args()

    rag = RAGPipeline(
        top_k=args.top_k,
        load_llm=True,
    )

    result = rag.build_index_from_path(
        input_path=args.data_dir,
        use_ocr_for_all_pdfs=args.ocr_all_pdfs,
        include_video_frame_ocr=not args.no_frame_ocr,
        video_frame_interval_sec=args.frame_interval,
    )

    print("\nIndex ready.")
    print(f"Indexed files: {result.total_files()}")
    print(f"Indexed chunks: {result.total_chunks()}")

    print("\nCommands:")
    print("- files")
    print("- sources")
    print("- summary")
    print("- quiz prep")
    print("- quiz")
    print("- exam")
    print("- flashcards")
    print("- pdf:<question>")
    print("- ocrpdf:<question>")
    print("- video:<question>")
    print("- frame:<question>")
    print("- image:<question>")
    print("- source=<text> :: <question>")
    print("- exit")
    print()

    last_question: str | None = None
    last_answer: str | None = None

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        effective_question = question

        # Basic follow-up support for short questions like:
        # "can you make it a table?"
        # "explain that more"
        if last_question and len(question.split()) <= 8:
            effective_question = (
                f"Previous question: {last_question}\n"
                f"Previous answer: {last_answer}\n"
                f"Follow-up question: {question}"
            )

        answer = route_question(rag, effective_question)

        print_block("Answer:", answer)

        last_question = question
        last_answer = answer


if __name__ == "__main__":
    main()