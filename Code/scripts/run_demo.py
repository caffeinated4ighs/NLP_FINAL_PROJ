from Code.scripts.RAGPipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    rag.build_index("/home/ubuntu/NLP_FINAL_PROJ/data/sample.pdf")

    print("\n--- ANSWER ---")
    print(rag.answer("What is this project about?"))

    print("\n--- SUMMARY ---")
    print(rag.summarize())

    print("\n--- QUIZ ---")
    print(rag.generate_quiz())


if __name__ == "__main__":
    main()