import asyncio
from src.question_gen import generate_questions
from src.semantic_eval import semantic_score
from src.llm_judge import judge_answer

class ExaminerState:
    def __init__(self):
        self.covered = set()
        self.weak = set()
        self.history = []
        self.count = 0
        self.max_q = 10

    def update(self, result):
        self.covered |= set(result["covered"])
        self.weak |= set(result["weak"])
        self.history.append(result)
        self.count += 1

    def done(self):
        return self.count >= self.max_q


class CognitiveExaminer:

    def __init__(self, memory_chunks):
        self.memory = memory_chunks
        self.state = ExaminerState()

    async def next_question(self):
        # Adaptive probing: weak concepts first
        context = " ".join(self.state.weak) if self.state.weak else " ".join(self.memory[:3])
        q = generate_questions(context)
        return q[0]   # one question at a time

    async def process_answer(self, q, student_answer):
        sem = semantic_score(student_answer, q["model_answer"])
        judge = await asyncio.to_thread(judge_answer, student_answer, q["expected_concepts"], q["model_answer"])
        judge = eval(judge)

        covered = [c for c in q["expected_concepts"] if c in student_answer.lower()]
        weak = list(set(q["expected_concepts"]) - set(covered))

        result = {
            "covered": covered,
            "weak": weak,
            "score": sem * 0.4 + judge["coverage"] * 0.6
        }

        self.state.update(result)
        return result
