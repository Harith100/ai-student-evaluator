from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL
import json

client = Groq(api_key=GROQ_API_KEY)

def generate_questions(context: str):
    prompt = f"""
You are a CBSE Class 10 Biology teacher.

Return ONLY a valid JSON array.
No markdown. No explanation.

Format:
[
  {{
    "id": 1,
    "question": "...",
    "model_answer": "...",
    "expected_concepts": ["..."]
  }}
]

CONTENT:
{context}
"""
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return res.choices[0].message.content.strip()
