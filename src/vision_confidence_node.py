import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

class VisionConfidenceNode:
    def __init__(self):
        self.blinks = 0
        self.freeze_events = 0
        self.last_landmarks = None
        self.history = deque(maxlen=120)
        self.start = time.time()

    def start_stream(self):
        cap = cv2.VideoCapture(0)
        with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mesh.process(rgb)

                if res.multi_face_landmarks:
                    lm = np.array([(p.x, p.y) for p in res.multi_face_landmarks[0].landmark])

                    left_eye = lm[LEFT_EYE]
                    right_eye = lm[RIGHT_EYE]

                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                    self.history.append(ear)

                    # Blink detection
                    if ear < 0.18:
                        self.blinks += 1

                    # Micro-freeze detection
                    if self.last_landmarks is not None:
                        delta = np.mean(np.abs(lm - self.last_landmarks))
                        if delta < 0.0004:
                            self.freeze_events += 1

                    self.last_landmarks = lm

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()

    def confidence_score(self):
        blink_rate = self.blinks / max((time.time() - self.start), 1)
        freeze_penalty = min(self.freeze_events / 50, 1)

        score = max(0, 1 - blink_rate * 0.15 - freeze_penalty * 0.4)
        return {
            "blink_rate": round(blink_rate, 2),
            "freeze_events": self.freeze_events,
            "confidence_score": round(score, 3)
        }
