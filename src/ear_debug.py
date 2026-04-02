"""
EAR Debug Script
Run this, watch the EAR value while blinking.
Open eyes = high value, closed eyes = low value.
That tells us exactly what threshold to set.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import os

RIGHT = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT  = [362, 385, 386, 387, 263, 373, 374, 380]

def ear_3d(pts):
    P0,P3,P4,P5,P8,P11,P12,P13 = pts
    n = (np.linalg.norm(P3-P13)**3 +
         np.linalg.norm(P4-P12)**3 +
         np.linalg.norm(P5-P11)**3)
    d = 3 * np.linalg.norm(P0-P8)**3
    return n / (d + 1e-6)

# ── Setup landmarker in LIVE_STREAM mode for webcam ──
latest_result = {"lm": None}

def result_callback(result, output_image, timestamp_ms):
    if result.face_landmarks:
        latest_result["lm"] = result.face_landmarks[0]
    else:
        latest_result["lm"] = None

model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_face_blendshapes=False,
    num_faces=1,
    result_callback=result_callback
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ts  = 0
ear_min = 9999.0
ear_max = 0.0

print("Watching EAR values. Blink normally, then blink hard.")
print("Press Q to quit and see your min/max summary.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_img, ts)
    ts += 33

    lm = latest_result["lm"]
    if lm:
        pts = np.array([[l.x, l.y, l.z] for l in lm])
        r   = ear_3d(pts[RIGHT])
        l_  = ear_3d(pts[LEFT])
        ear = (r + l_ + 1) / 2

        ear_min = min(ear_min, ear)
        ear_max = max(ear_max, ear)

        # Color: green = above 0.45, red = below (currently "closed")
        color = (0, 220, 80) if ear > 0.45 else (0, 60, 255)

        cv2.putText(frame, f"EAR:  {ear:.4f}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
        cv2.putText(frame, f"R: {r:.3f}   L: {l_:.3f}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(frame, f"min: {ear_min:.4f}  max: {ear_max:.4f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 255), 1)
        cv2.putText(frame, "BLINK NOW  ->  watch EAR drop", (30, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    else:
        cv2.putText(frame, "No face detected", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 255), 2)

    cv2.imshow("EAR Debug — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print(f"\n── EAR Summary ──────────────────")
print(f"  Eyes OPEN  (max): {ear_max:.4f}")
print(f"  Eyes CLOSED (min): {ear_min:.4f}")
print(f"  Suggested threshold: {(ear_max + ear_min) / 2:.4f}")
print(f"─────────────────────────────────")