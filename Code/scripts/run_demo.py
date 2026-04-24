from backend.rag_pipeline import RAGPipeline

rag = RAGPipeline()

rag.build_index("/home/ubuntu/NLP_FINAL_PROJ/data/sample.pdf")

print("\nANSWER:\n")
print(rag.answer("What is this document about?"))