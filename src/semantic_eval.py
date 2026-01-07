from sentence_transformers import SentenceTransformer, util
from src.fake_answer_generator import generate_fake_answers

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_score(student, teacher):
    if not student.strip():
        return 0.0

    fake_answers = generate_fake_answers(teacher, student)

    teacher_emb = model.encode(teacher, convert_to_tensor=True, normalize_embeddings=True)
    student_emb = model.encode(student, convert_to_tensor=True, normalize_embeddings=True)
    fake_embs = model.encode(fake_answers, convert_to_tensor=True, normalize_embeddings=True)

    st = util.cos_sim(student_emb, teacher_emb).item()
    fake_sims = util.cos_sim(student_emb, fake_embs).cpu().numpy()[0]
    max_fake = float(max(fake_sims))

    normalized = st / (1 + max_fake)

    α, β, γ = 1.0, 0.3, 0.4
    factor = α*st - β*max_fake + γ*normalized

    return max(0.0, min(1.0, factor))
