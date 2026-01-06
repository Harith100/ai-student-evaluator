import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_score(student_answer: str, model_answer: str):
    s_emb = model.encode(student_answer)
    m_emb = model.encode(model_answer)
    return float(cosine_similarity(s_emb, m_emb))
