from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = "Generate 5 incorrect but plausible answers to confuse students. Return ONLY quoted sentences."

def generate_fake_answers(teacher, student):
    msg = f'Teacher Answer: "{teacher}"\nStudent Answer: "{student}"'
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":msg}
        ],
        temperature=0.7
    )

    import re
    return re.findall(r'"([^"]+)"', res.choices[0].message.content)
