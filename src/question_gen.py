from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

def generate_questions(context: str):
    prompt = f"""
You are a highly experienced subject expert and university-level professor.

You are given study material from any academic domain (science, engineering, medicine, law, business, humanities, etc.).

Your task is to carefully analyze the material and create rigorous, concept-focused exam questions that test real understanding, reasoning, and applied knowledge.

Return ONLY a valid JSON array.
No markdown. No explanation.

Each object must strictly follow this format:

[
  {{
    "id": 1,
    "question": "...",
    "model_answer": "...",
    "expected_concepts": ["..."]
  }}
]

Rules:
- Questions must be academically sound, unambiguous, and based strictly on the provided content.
- Avoid opinion-based or vague questions.
- Model answers must be precise, correct, and concise.
- Expected concepts must list the essential ideas that must appear in a correct answer.
- Generate between 8 and 12 high-quality questions depending on content depth.

STUDY MATERIAL:
{context}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()
