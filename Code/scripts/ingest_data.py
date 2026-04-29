from Code.scripts.rag_pipeline import RAGPipeline


def print_block(title: str, body: str) -> None:
    print(f"\n{title}")
    print(body)
    print("\n" + "-" * 80 + "\n")


def main():
    rag = RAGPipeline(
        top_k=12,
        use_chroma=False,  # Set True only after vector_store.py + chromadb are stable.
    )

    rag.build_index_from_data_folder(
        data_dir="data",
        use_ocr_for_all_pdfs=False,
        video_frame_interval_sec=10,
    )

    print("\nIndex ready. Ask questions about your files.")
    print("Commands:")
    print("- files")
    print("- sources")
    print("- quiz prep")
    print("- pdf:<question>")
    print("- video:<question>")
    print("- image:<question>")
    print("- source=<text> :: <question>")
    print("- exit")
    print()

    last_question = None
    last_answer = None

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        q_lower = question.lower()

        if q_lower in {"files", "list files", "indexed files", "what files are indexed"}:
            print_block("Indexed files:", rag.list_uploaded_files())
            continue

        if q_lower in {"sources", "source summary", "context sources"}:
            print_block("Indexed source summary:", rag.get_source_summary_table())
            continue

        if q_lower in {"quiz prep", "summarize each file", "break down each file"}:
            print_block("Per-source quiz prep:", rag.summarize_each_source())
            continue

        if q_lower.startswith("pdf:"):
            actual_question = question.split(":", 1)[1].strip()
            answer = rag.answer_filtered(
                actual_question,
                modality="pdf_text",
                top_k=12,
            )
            print_block("Answer:", answer)
            continue

        if q_lower.startswith("video:"):
            actual_question = question.split(":", 1)[1].strip()
            answer = rag.answer_filtered(
                actual_question,
                modality="video_transcript",
                top_k=12,
            )
            print_block("Answer:", answer)
            continue

        if q_lower.startswith("image:"):
            actual_question = question.split(":", 1)[1].strip()
            answer = rag.answer_filtered(
                actual_question,
                modality="image_ocr",
                top_k=12,
            )
            print_block("Answer:", answer)
            continue

        if q_lower.startswith("source=") and "::" in question:
            source_filter, actual_question = question.split("::", 1)
            source_filter = source_filter.replace("source=", "").strip()
            actual_question = actual_question.strip()

            answer = rag.answer_filtered(
                actual_question,
                source_contains=source_filter,
                top_k=12,
            )
            print_block("Answer:", answer)
            continue

        effective_question = question

        # Basic follow-up support.
        # Example:
        # Q1: source=nlp_lecture5_1 :: give me the topics
        # Q2: can you put that in a table
        if last_question and len(question.split()) <= 8:
            effective_question = (
                f"Previous question: {last_question}\n"
                f"Previous answer: {last_answer}\n"
                f"Follow-up question: {question}"
            )

        answer = rag.answer(effective_question)
        print_block("Answer:", answer)

        last_question = question
        last_answer = answer


if __name__ == "__main__":
    main()