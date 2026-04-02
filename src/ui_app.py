import gradio as gr
import json

from src.pdf_loader import load_pdf
from src.chunker import chunk_text
from src.question_gen import generate_questions
from src.semantic_eval import semantic_score
from src.llm_judge import judge_answer


def generate(pdf_file):
    text = load_pdf(pdf_file.name)
    chunks = chunk_text(text)
    return json.loads(generate_questions(" ".join(chunks[:5])))


def fill_questions(questions):
    outputs = []
    for i in range(12):
        outputs.append(questions[i]["question"] if i < len(questions) else "")
    return outputs


def evaluate_answers(questions, *answers):
    results = []
    for i, q in enumerate(questions):
        student = answers[i].strip()

        if not student:
            results.append(f"Q{i+1}: 0.00/10 — No answer provided.")
            continue

        sem = semantic_score(student, q["model_answer"])
        judge = json.loads(judge_answer(student, q["expected_concepts"], q["model_answer"]))

        raw = sem * 0.40 + judge["coverage"] * 0.60

        final = max(0.0, min(1.0, raw))
        display = final * 10

        results.append(f"Q{i+1}: {display:.2f}/10 — {judge['feedback']}")

    return "\n".join(results)


with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Student Evaluator")

    with gr.Tab("📄 Upload"):
        pdf = gr.File(label="Upload Study PDF")
        gen_btn = gr.Button("Generate Exam")

    with gr.Tab("📝 Answer"):
        questions_state = gr.State()
        answer_boxes = []
        q_labels = []

        for i in range(12):
            q = gr.Markdown()
            a = gr.Textbox(lines=3, label=f"Your Answer {i+1}")
            q_labels.append(q)
            answer_boxes.append(a)

        eval_btn = gr.Button("Evaluate")
        result_box = gr.Textbox(lines=20, label="Results")


    gen_btn.click(generate, pdf, questions_state)\
           .then(fill_questions, questions_state, q_labels)

    eval_btn.click(evaluate_answers, [questions_state] + answer_boxes, result_box)

demo.launch()
