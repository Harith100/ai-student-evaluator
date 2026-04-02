# src/models.py
from dataclasses import dataclass

@dataclass
class QuestionResult:
    question: str
    answer: str
    score: float
    feedback: str
    audio_path: str
    video_path: str
    audio_conf: dict | None = None
    video_conf: dict | None = None
