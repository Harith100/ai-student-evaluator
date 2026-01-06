from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    id: str
    question: str
    model_answer: str
    expected_concepts: List[str]

class StudentAnswer(BaseModel):
    question_id: str
    answer: str
