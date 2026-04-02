"""
Confidence Analyzer — Final Production Version
===============================================
Calibrated config locked in (Clip A: 92.2, Clip B: 59.6)

Architecture:
  - MediaPipe FaceLandmarker (VIDEO mode)
  - EAR blink detection with consecutive-frame guard (Python-Gaze-Face-Tracker)
  - AngleBuffer head-pose smoother
  - Head-pose yaw/pitch gaze detection
  - Mamdani fuzzy inference (skfuzzy), 81 rules, 4 inputs → 1 output

Inputs:  face_ratio, head_stability, gaze_presence, blink_score
Output:  confidence (0.0 – 1.0)

Fine-tuning knobs still exposed in Flask UI:
  gaze thresholds, stability scale/window, EAR params, blink curve
MF boundaries are locked — edit LOCKED_MF in code only.
"""

import cv2
import time
import os
import collections
import tempfile

import numpy as np
import mediapipe as mp
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from flask import Flask, request, jsonify, send_from_directory
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

app = Flask(__name__, static_folder="static")

UPLOAD_DIR = tempfile.mkdtemp()
# ═══════════════════════════════════════════════════════════════════════════════
# AngleBuffer
# ═══════════════════════════════════════════════════════════════════════════════
class AngleBuffer:
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        return np.mean(self.buffer, axis=0) if self.buffer else np.zeros(3)


# ═══════════════════════════════════════════════════════════════════════════════
# EAR blink detection
# ═══════════════════════════════════════════════════════════════════════════════
RIGHT_EYE_EAR_IDX = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_EAR_IDX  = [362, 385, 386, 387, 263, 373, 374, 380]

def _euclidean_distance_3d(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    num = (np.linalg.norm(P3 - P13) ** 3 +
           np.linalg.norm(P4 - P12) ** 3 +
           np.linalg.norm(P5 - P11) ** 3)
    den = 3 * np.linalg.norm(P0 - P8) ** 3
    return num / (den + 1e-6)

def compute_ear(lm_3d):
    r = _euclidean_distance_3d(lm_3d[RIGHT_EYE_EAR_IDX])
    l = _euclidean_distance_3d(lm_3d[LEFT_EYE_EAR_IDX])
    return (r + l + 1) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# Head pose
# ═══════════════════════════════════════════════════════════════════════════════
_POSE_IDX = [1, 33, 61, 199, 263, 291]

def _estimate_head_pose(lm_2d_px, img_h, img_w):
    focal   = float(img_w)
    cam_mat = np.array([[focal, 0, img_h/2],
                        [0, focal, img_w/2],
                        [0, 0, 1]], dtype=np.float64)
    dist    = np.zeros((4, 1), dtype=np.float64)
    pts_2d  = lm_2d_px[_POSE_IDX].astype(np.float64)
    pts_3d  = np.hstack([pts_2d, np.zeros((len(_POSE_IDX), 1))])
    ok, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, cam_mat, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    return angles[0]*360, angles[1]*360, angles[2]*360


# ═══════════════════════════════════════════════════════════════════════════════
# LOCKED membership function boundaries  (92.2 / 59.6 calibration)
# Edit here in code only — not exposed in UI to prevent accidental breakage
# ═══════════════════════════════════════════════════════════════════════════════
LOCKED_MF = {
    # face_ratio
    "mf_face_low_a":   0.00, "mf_face_low_b":  0.00,
    "mf_face_low_c":   0.35, "mf_face_low_d":  0.55,
    "mf_face_med_a":   0.40, "mf_face_med_b":  0.60, "mf_face_med_c": 0.80,
    "mf_face_high_a":  0.65, "mf_face_high_b": 0.85,
    "mf_face_high_c":  1.00, "mf_face_high_d": 1.00,

    # head_stability  ← tightened to push 0.575 into "low"
    "mf_stab_low_a":   0.00, "mf_stab_low_b":  0.00,
    "mf_stab_low_c":   0.55, "mf_stab_low_d":  0.68,
    "mf_stab_med_a":   0.58, "mf_stab_med_b":  0.72, "mf_stab_med_c": 0.86,
    "mf_stab_high_a":  0.75, "mf_stab_high_b": 0.92,
    "mf_stab_high_c":  1.00, "mf_stab_high_d": 1.00,

    # gaze_presence  ← tightened but med_a < med_b < med_c preserved
    "mf_gaze_low_a":   0.00, "mf_gaze_low_b":  0.00,
    "mf_gaze_low_c":   0.45, "mf_gaze_low_d":  0.62,
    "mf_gaze_med_a":   0.48, "mf_gaze_med_b":  0.65, "mf_gaze_med_c": 0.82,
    "mf_gaze_high_a":  0.70, "mf_gaze_high_b": 0.88,
    "mf_gaze_high_c":  1.00, "mf_gaze_high_d": 1.00,

    # blink_score
    "mf_blink_low_a":  0.00, "mf_blink_low_b": 0.00,
    "mf_blink_low_c":  0.30, "mf_blink_low_d": 0.50,
    "mf_blink_med_a":  0.35, "mf_blink_med_b": 0.55, "mf_blink_med_c": 0.75,
    "mf_blink_high_a": 0.60, "mf_blink_high_b": 0.80,
    "mf_blink_high_c": 1.00, "mf_blink_high_d": 1.00,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tunable detection parameters (still exposed in UI)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS = {
    # Gaze
    "gaze_yaw_threshold":    8.0,
    "gaze_pitch_threshold":  8.0,

    # Head stability
    "stability_window":      20,
    "stability_exp_scale":   12.0,
    "angle_smooth_window":   10,

    # EAR blink
    "ear_threshold": 0.59,   
    "ear_consec_frames": 2,

    # Blink score curve
    "blink_ideal": 24.0,   # matched to your actual rate
    "blink_sigma":  8.0,   # wider tolerance so small variation doesn't hurt score
    "blink_max_reasonable":   35.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MF validation — catches broken triangle ordering before skfuzzy silently fails
# ═══════════════════════════════════════════════════════════════════════════════
def _validate_mf(mf):
    errors = []
    for prefix in ('face', 'stab', 'gaze', 'blink'):
        la, lb, lc, ld = (mf[f"mf_{prefix}_low_{x}"]  for x in ('a','b','c','d'))
        ma, mb, mc     = (mf[f"mf_{prefix}_med_{x}"]  for x in ('a','b','c'))
        ha, hb, hc, hd = (mf[f"mf_{prefix}_high_{x}"] for x in ('a','b','c','d'))

        if not (la <= lb <= lc <= ld):
            errors.append(f"{prefix} LOW trapezoid not ordered: {la},{lb},{lc},{ld}")
        if not (ma <= mb <= mc):
            errors.append(f"{prefix} MED triangle not ordered: {ma},{mb},{mc}")
        if not (ha <= hb <= hc <= hd):
            errors.append(f"{prefix} HIGH trapezoid not ordered: {ha},{hb},{hc},{hd}")
        if ld > mb:
            errors.append(f"{prefix} LOW/MED overlap issue: low_d={ld} > med_b={mb}")
        if mc > hb:
            errors.append(f"{prefix} MED/HIGH overlap issue: med_c={mc} > high_b={hb}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Fuzzy system builder
# ═══════════════════════════════════════════════════════════════════════════════
def _build_fuzzy_system(mf):
    u = np.linspace(0, 1, 200)

    face  = ctrl.Antecedent(u, 'face_ratio')
    stab  = ctrl.Antecedent(u, 'head_stability')
    gaze  = ctrl.Antecedent(u, 'gaze_presence')
    blink = ctrl.Antecedent(u, 'blink_score')
    conf  = ctrl.Consequent(u, 'confidence', defuzzify_method='centroid')

    def set_mfs(var, prefix):
        var['low']  = fuzz.trapmf(u, [mf[f"mf_{prefix}_low_a"],  mf[f"mf_{prefix}_low_b"],
                                       mf[f"mf_{prefix}_low_c"],  mf[f"mf_{prefix}_low_d"]])
        var['med']  = fuzz.trimf (u, [mf[f"mf_{prefix}_med_a"],  mf[f"mf_{prefix}_med_b"],
                                       mf[f"mf_{prefix}_med_c"]])
        var['high'] = fuzz.trapmf(u, [mf[f"mf_{prefix}_high_a"], mf[f"mf_{prefix}_high_b"],
                                       mf[f"mf_{prefix}_high_c"], mf[f"mf_{prefix}_high_d"]])

    set_mfs(face,  'face')
    set_mfs(stab,  'stab')
    set_mfs(gaze,  'gaze')
    set_mfs(blink, 'blink')

    conf['very_low']  = fuzz.trapmf(u, [0.00, 0.00, 0.10, 0.20])
    conf['low']       = fuzz.trimf (u, [0.10, 0.25, 0.40])
    conf['medium']    = fuzz.trimf (u, [0.30, 0.50, 0.70])
    conf['high']      = fuzz.trimf (u, [0.60, 0.75, 0.90])
    conf['very_high'] = fuzz.trapmf(u, [0.80, 0.90, 1.00, 1.00])

    # All 81 rules
    lvl_map = {'low': 0, 'med': 1, 'high': 2}

    def rule_output(f, s, g, b):
        pf, ps, pg, pb = lvl_map[f], lvl_map[s], lvl_map[g], lvl_map[b]
        # Hard overrides
        if pg == 0 and ps == 0:               return 'very_low'
        if pg == 2 and ps == 2 and pb >= 1 and pf >= 1: return 'very_high'
        # Graded: primary (gaze+stab) weighted 2x
        total = pg*2 + ps*2 + pf + pb         # range 0–10
        if total <= 2:  return 'very_low'
        if total <= 4:  return 'low'
        if total <= 6:  return 'medium'
        if total <= 8:  return 'high'
        return 'very_high'

    rules = []
    for f in ('low', 'med', 'high'):
        for s in ('low', 'med', 'high'):
            for g in ('low', 'med', 'high'):
                for b in ('low', 'med', 'high'):
                    rules.append(ctrl.Rule(
                        face[f] & stab[s] & gaze[g] & blink[b],
                        conf[rule_output(f, s, g, b)]
                    ))

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


# ═══════════════════════════════════════════════════════════════════════════════
# Core analysis
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_video(video_path, params):
    p  = {**DEFAULT_PARAMS, **params}
    mf = LOCKED_MF   # always use locked MF boundaries

    # Validate MF (safety check)
    errs = _validate_mf(mf)
    if errs:
        return {"error": "MF validation failed: " + "; ".join(errs), "confidence": 0.0}

    model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    start_ts = 0

    total_frames     = 0
    face_frames      = 0
    gaze_away_frames = 0
    nose_x_history   = []
    stability_windows= []
    stab_win         = int(p["stability_window"])
    angle_buf        = AngleBuffer(size=int(p["angle_smooth_window"]))
    blink_count      = 0
    ear_counter      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        img_h, img_w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_img, int(start_ts))
        start_ts += int(1000 / fps)

        if not result.face_landmarks:
            ear_counter = 0
            continue

        face_frames += 1
        lm       = result.face_landmarks[0]
        lm_3d    = np.array([[l.x, l.y, l.z] for l in lm])
        lm_2d_px = np.array([[l.x * img_w, l.y * img_h] for l in lm])

        # Head stability
        nose_x = lm[1].x
        nose_x_history.append(nose_x)
        if len(nose_x_history) >= stab_win:
            stability_windows.append(np.std(nose_x_history[-stab_win:]))

        # Gaze via head pose
        pitch, yaw, roll = _estimate_head_pose(lm_2d_px, img_h, img_w)
        angle_buf.add([pitch, yaw, roll])
        s_pitch, s_yaw, _ = angle_buf.get_average()
        if (abs(s_yaw) > p["gaze_yaw_threshold"] or
                abs(s_pitch) > p["gaze_pitch_threshold"]):
            gaze_away_frames += 1

        # EAR blink
        ear = compute_ear(lm_3d)
        if ear <= p["ear_threshold"]:
            ear_counter += 1
        else:
            if ear_counter >= int(p["ear_consec_frames"]):
                blink_count += 1
            ear_counter = 0

    cap.release()
    landmarker.close()

    if total_frames == 0 or face_frames == 0:
        return {"confidence": 0.0, "face_ratio": 0.0, "head_stability": 0.0,
                "gaze_presence": 0.0, "blink_rate": 0.0, "blink_score": 0.0,
                "frames_analyzed": int(total_frames), "error": "No face detected"}

    # Raw metrics
    face_ratio     = float(np.clip(face_frames / total_frames, 0, 1))
    raw_stab       = np.mean(stability_windows) if stability_windows else 0.0
    head_stability = float(np.clip(np.exp(-raw_stab * p["stability_exp_scale"]), 0, 1))
    gaze_presence  = float(np.clip(1.0 - (gaze_away_frames / face_frames), 0, 1))
    duration_min   = (total_frames / fps) / 60.0
    blink_rate     = min(blink_count / max(duration_min, 1e-6), p["blink_max_reasonable"])
    blink_score    = float(np.clip(
        np.exp(-((blink_rate - p["blink_ideal"]) ** 2) / (2 * p["blink_sigma"] ** 2)),
        0, 1))

    # Fuzzy inference
    try:
        sim = _build_fuzzy_system(mf)
        sim.input['face_ratio']     = face_ratio
        sim.input['head_stability'] = head_stability
        sim.input['gaze_presence']  = gaze_presence
        sim.input['blink_score']    = blink_score
        sim.compute()
        confidence = float(np.clip(sim.output['confidence'], 0, 1))
    except Exception as e:
        # Fallback: equal-weight average
        confidence = float(np.clip(
            (face_ratio + head_stability + gaze_presence + blink_score) / 4, 0, 1))

    return {
        "confidence":      round(confidence,      3),
        "face_ratio":      round(face_ratio,       3),
        "head_stability":  round(head_stability,   3),
        "gaze_presence":   round(gaze_presence,    3),
        "blink_rate":      round(blink_rate,       2),
        "blink_score":     round(blink_score,      3),
        "frames_analyzed": int(total_frames),
        "params_used":     {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in p.items()}
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Flask routes
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/defaults")
def defaults():
    return jsonify(DEFAULT_PARAMS)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    f    = request.files["video"]
    path = os.path.join(UPLOAD_DIR, "uploaded.mp4")
    f.save(path)
    return jsonify({"path": path, "message": "Uploaded OK"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data       = request.json or {}
    video_path = data.get("video_path")
    params     = data.get("params", {})
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video not found. Upload or record first."}), 400
    return jsonify(analyze_video(video_path, params))

@app.route("/record", methods=["POST"])
def record():
    data     = request.json or {}
    duration = int(data.get("duration", 8))
    out_path = os.path.join(UPLOAD_DIR, "recorded.mp4")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open webcam"}), 500
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return jsonify({"path": out_path, "message": f"Recorded {duration}s"})


if __name__ == "__main__":
    app.run(debug=True, port=5055)