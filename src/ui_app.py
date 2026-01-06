import gradio as gr
import json, re
from bs4 import BeautifulSoup

from src.pdf_loader import load_pdf
from src.chunker import chunk_text
from src.question_gen import generate_questions
from src.semantic_eval import semantic_score
from src.llm_judge import judge_answer

def safe_json(text):
    try:
        return json.loads(text)
    except:
        import re
        match = re.search(r'\{.*\}', text, re.S)
        return json.loads(match.group()) if match else {"coverage":0,"hallucination":1,"feedback":"Unable to parse answer"}

# ---------------- UTIL ----------------

def extract_json(text):
    match = re.search(r'\[.*\]', text, re.S)
    return match.group() if match else "[]"

# ---------------- GENERATE ----------------

def generate(pdf_file):
    text = load_pdf(pdf_file.name)
    chunks = chunk_text(text)
    raw = generate_questions(" ".join(chunks[:6]))
    questions = json.loads(extract_json(raw))
    return questions

def render_exam(questions):
    html = ""
    for i, q in enumerate(questions):
        html += f"""
        <div style="margin-bottom:20px;">
            <b>Q{i+1}. {q['question']}</b><br>
            <textarea rows="4" style="width:100%;"></textarea>
        </div>
        """
    return html

# ---------------- EVALUATE ----------------

def evaluate_html(questions, html):
    soup = BeautifulSoup(html, "html.parser")
    answers = [t.text.strip() for t in soup.find_all("textarea")]

    results = []
    for i, q in enumerate(questions):
        student = answers[i] if i < len(answers) else ""
        sem = semantic_score(student, q["model_answer"])
        judge = json.loads(judge_answer(student, q["expected_concepts"], q["model_answer"]))
        final = round((judge["coverage"]*0.5 + sem*0.35 + (1-judge["hallucination"])*0.20) * 10, 2)

        results.append(f"""
Q{i+1}: {q['question']}
Score: {final}/10
Feedback: {judge['feedback']}
""")

    return "\n".join(results)

# ---------------- UI ----------------

with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Student Evaluator")

    with gr.Tab("📄 Upload"):
        pdf = gr.File(label="Upload Study PDF")
        gen_btn = gr.Button("Generate Exam")

    with gr.Tab("📝 Answer"):
        questions_state = gr.State()
        exam_html = gr.HTML()
        eval_btn = gr.Button("Evaluate")
        result_box = gr.Textbox(lines=20, label="Results")

    gen_btn.click(generate, pdf, questions_state)\
           .then(render_exam, questions_state, exam_html)

    eval_btn.click(evaluate_html, [questions_state, exam_html], result_box)

demo.launch()
