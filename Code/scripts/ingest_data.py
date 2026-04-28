from Code.scripts.rag_pipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    rag.build_index_from_data_folder(
        data_dir="data",
        use_ocr_for_all_pdfs=False,
        video_frame_interval_sec=10,
    )

    print("\nIndex ready. Ask questions about your files.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        answer = rag.answer(question)
        print("\nAnswer:")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()