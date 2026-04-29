import os
import sys
import gradio as gr

# Allows frontend/app.py to import backend/rag_pipeline.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from scripts.RAGPipeline import RAGPipeline

rag = None


def load_model():
    global rag
    if rag is None:
        rag = RAGPipeline()
    return rag


def upload_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF first."

    pipeline = load_model()
    pipeline.build_index(pdf_file.name)

    return "PDF uploaded and indexed successfully."


def ask_question(question):
    if rag is None or rag.index is None:
        return "Please upload and index a PDF first."

    if not question.strip():
        return "Please enter a question."

    return rag.answer(question)


def summarize_pdf():
    if rag is None or rag.index is None:
        return "Please upload and index a PDF first."

    context = "\n\n".join(rag.chunks[:8])

    prompt = f"""
Summarize this study material clearly.

Include:
1. Main idea
2. Key concepts
3. Important details

Content:
{context}
"""

    return rag.generate(prompt)


def generate_quiz():
    if rag is None or rag.index is None:
        return "Please upload and index a PDF first."

    context = "\n\n".join(rag.chunks[:8])

    prompt = f"""
Create 5 multiple choice quiz questions from this study material.
Each question should include 4 options and the correct answer.

Content:
{context}
"""

    return rag.generate(prompt)


with gr.Blocks(title="RAG Study Assistant") as app:
    gr.Markdown("# RAG-Based Study Assistant")
    gr.Markdown("Upload a PDF, ask questions, generate summaries, and create quizzes.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Status")

    upload_button = gr.Button("Index PDF")
    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_input,
        outputs=upload_status
    )

    gr.Markdown("## Ask Questions")
    question_input = gr.Textbox(label="Question", placeholder="What is this document about?")
    answer_output = gr.Textbox(label="Answer", lines=8)

    ask_button = gr.Button("Ask")
    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

    gr.Markdown("## Study Tools")

    with gr.Row():
        summary_button = gr.Button("Generate Summary")
        quiz_button = gr.Button("Generate Quiz")

    summary_output = gr.Textbox(label="Summary", lines=10)
    quiz_output = gr.Textbox(label="Quiz", lines=12)

    summary_button.click(
        fn=summarize_pdf,
        inputs=None,
        outputs=summary_output
    )

    quiz_button.click(
        fn=generate_quiz,
        inputs=None,
        outputs=quiz_output
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)