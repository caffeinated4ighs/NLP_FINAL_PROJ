# LearnLoop: NLP Final Project

LearnLoop is a multimodal RAG-based study assistant. The system takes course materials such as PDFs, images, and videos, converts them into searchable text chunks, retrieves the most relevant chunks for a user query, and uses an LLM to generate study outputs such as answers, summaries, quizzes, flashcards, and exam-style questions.

The code should be run in the following order:

1.⁠ ⁠Set up the environment
2.⁠ ⁠Add raw input files to the ⁠ data/ ⁠ folder
3.⁠ ⁠Run ingestion and preprocessing
4.⁠ ⁠Build embeddings and the retrieval index/vector database
5.⁠ ⁠Run the RAG pipeline/modeling code
6.⁠ ⁠Run the backend API
7.⁠ ⁠Run the frontend application 

## Recommended full run order: 

# 1. Install Python dependencies
uv sync

# 2. Add files to data/
# Example: data/lecture.pdf or data/video.mp4

# 3. Test ingestion, indexing, retrieval, and generation
uv run python -c "from Code.rag.rag_pipeline import RAGPipeline; rag = RAGPipeline(); rag.build_index_from_path('data'); print(rag.answer('What are the main topics?'))"

# 4. Start the backend API
uv run python Code/api/run_server.py

# 5. Start the frontend
cd Code/frontend
npm install
npm run dev
