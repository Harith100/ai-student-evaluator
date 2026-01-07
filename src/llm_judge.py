from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL
import json

client = Groq(api_key=GROQ_API_KEY)

def judge_answer(student, expected_concepts, reference_answer):

    # HARD ZERO — prevents hallucinated grading
    if not student.strip():
        return json.dumps({
            "coverage": 0.0,
            "hallucination": 0.0,
            "feedback": "No answer provided."
        })

    prompt = f"""
You are a senior subject professor evaluating conceptual understanding.

You MUST judge by MEANING, not by keyword presence.
If the idea is correctly conveyed, it is FULLY correct even if wording differs.

DO NOT penalize for missing exact words.
DO NOT penalize for short answers if they are correct.

Return ONLY valid JSON.
No markdown. No explanation.

Format:
{{
  "coverage": 0.0,
  "hallucination": 0.0,
  "feedback": ""
}}

Student Answer:
{student}

Expected Concepts:
{expected_concepts}

Reference Answer:
{reference_answer}

Rules:
- coverage = fraction of concepts clearly understood (0–1)
- hallucination = fraction of clearly incorrect ideas (0–1)
- feedback must be brief, fair, and human-like
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
