from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL
import json

client = Groq(api_key=GROQ_API_KEY)

def judge_answer(student, expected_concepts, source_text):
    prompt = f"""
You are a strict CBSE biology examiner.

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
{source_text}
"""
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content.strip()
