
import json
from multiprocessing import context
from src.question_gen import generate_questions
from src.semantic_eval import semantic_score
from src.llm_judge import judge_answer
from src.json_sanitizer import safe_json_array
from src.topic_mapper import build_topic_map
from src.json_sanitizer import safe_json_array


class ExaminerState:
    def __init__(self):
        self.covered = set()
        self.weak = set()
        self.history = []
        self.count = 0
        self.max_q = 4 # max questions per exam

    def update(self, result):
        self.covered |= set(result["covered"])
        self.weak |= set(result["weak"])
        self.history.append(result)
        self.count += 1

    def done(self):
        return self.count >= self.max_q


class AdaptiveExaminer:

    def __init__(self, memory_chunks):
        self.memory = memory_chunks
        self.state = ExaminerState()

        # Build syllabus topic map once
        self.topic_map = build_topic_map(memory_chunks)
        self.uncovered_topics = set(self.topic_map.keys())

    async def ask_next(self):

     # Phase-1: syllabus coverage
     if self.uncovered_topics:
         topic = self.uncovered_topics.pop()
         context = " ".join(self.topic_map[topic][:3])

     # Phase-2: probe weaknesses
     elif self.state.weak:
         context = " ".join(self.state.weak)

     else:
         context = " ".join(self.memory[:3])

     raw = generate_questions(context)
     return safe_json_array(raw)[0]

    
   # One question at a time

    async def grade(self, q, student_answer):
        sem = semantic_score(student_answer, q["model_answer"])
        judge = json.loads(judge_answer(student_answer, q["expected_concepts"], q["model_answer"]))
    
        covered = [c for c in q["expected_concepts"] if c.lower() in student_answer.lower()]
        weak = list(set(q["expected_concepts"]) - set(covered))
    
        result = {
            "score": sem * 0.4 + judge["coverage"] * 0.6,
            "covered": covered,
            "weak": weak,
            "feedback": judge["feedback"]
        }
    
        # ---- NEW: mark syllabus topic as covered ----
        for t, chunks in self.topic_map.items():
            if any(chunk in q["model_answer"] for chunk in chunks):
                self.uncovered_topics.discard(t)
        # ---------------------------------------------
    
        self.state.update(result)
        return result

