from src.pdf_loader import load_pdf
from src.chunker import chunk_text
from src.memory import build_memory
from src.question_gen import generate_questions
from fastapi import FastAPI
app = FastAPI(title="Student Evaluator v1")

@app.post("/ingest")
def ingest(pdf_path: str):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    index, memory = build_memory(chunks)
    questions = generate_questions(" ".join(chunks[:5]))
    return {"status": "ok", "questions": questions}
