import cv2
import time
import collections
import numpy as np
import mediapipe as mp
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ═══════════════════════════════════════════════════════════════════════════════
# AngleBuffer — smooths head-pose angles over a rolling window
# ═══════════════════════════════════════════════════════════════════════════════
class _AngleBuffer:
    def __init__(self, size=10):
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        return np.mean(self.buffer, axis=0) if self.buffer else np.zeros(3)


# ═══════════════════════════════════════════════════════════════════════════════
# EAR blink detection  (Python-Gaze-Face-Tracker / Asadullah Dal)
# ═══════════════════════════════════════════════════════════════════════════════
_RIGHT_EYE = [33,  160, 159, 158, 133, 153, 145, 144]
_LEFT_EYE  = [362, 385, 386, 387, 263, 373, 374, 380]

def _ear_3d(pts):
    P0,P3,P4,P5,P8,P11,P12,P13 = pts
    n = (np.linalg.norm(P3-P13)**3 +
         np.linalg.norm(P4-P12)**3 +
         np.linalg.norm(P5-P11)**3)
    d = 3 * np.linalg.norm(P0-P8)**3
    return n / (d + 1e-6)

def _compute_ear(lm_3d):
    r = _ear_3d(lm_3d[_RIGHT_EYE])
    l = _ear_3d(lm_3d[_LEFT_EYE])
    return (r + l + 1) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# Head pose estimation
# ═══════════════════════════════════════════════════════════════════════════════
_POSE_IDX = [1, 33, 61, 199, 263, 291]

def _head_pose(lm_2d_px, img_h, img_w):
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
# Locked calibration config  (Clip A: 92.2  Clip B: 59.6)
# ═══════════════════════════════════════════════════════════════════════════════
_PARAMS = {
    "gaze_yaw_threshold":    8.0,
    "gaze_pitch_threshold":  8.0,
    "stability_window":      20,
    "stability_exp_scale":   12.0,
    "angle_smooth_window":   10,
    "ear_threshold":         0.59,   # calibrated to user's eyes via ear_debug.py
    "ear_consec_frames":      2,
    "blink_ideal":           24.0,   # matched to user's natural rate
    "blink_sigma":            8.0,
    "blink_max_reasonable":  45.0,
}

_MF = {
    # face_ratio
    "mf_face_low_a":   0.00, "mf_face_low_b":  0.00,
    "mf_face_low_c":   0.35, "mf_face_low_d":  0.55,
    "mf_face_med_a":   0.40, "mf_face_med_b":  0.60, "mf_face_med_c": 0.80,
    "mf_face_high_a":  0.65, "mf_face_high_b": 0.85,
    "mf_face_high_c":  1.00, "mf_face_high_d": 1.00,
    # head_stability
    "mf_stab_low_a":   0.00, "mf_stab_low_b":  0.00,
    "mf_stab_low_c":   0.55, "mf_stab_low_d":  0.68,
    "mf_stab_med_a":   0.58, "mf_stab_med_b":  0.72, "mf_stab_med_c": 0.86,
    "mf_stab_high_a":  0.75, "mf_stab_high_b": 0.92,
    "mf_stab_high_c":  1.00, "mf_stab_high_d": 1.00,
    # gaze_presence
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
# Fuzzy system  (Mamdani, 81 rules)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_fuzzy_system():
    u    = np.linspace(0, 1, 200)
    mf   = _MF

    face  = ctrl.Antecedent(u, 'face_ratio')
    stab  = ctrl.Antecedent(u, 'head_stability')
    gaze  = ctrl.Antecedent(u, 'gaze_presence')
    blink = ctrl.Antecedent(u, 'blink_score')
    conf  = ctrl.Consequent(u, 'confidence', defuzzify_method='centroid')

    def mfs(var, px):
        var['low']  = fuzz.trapmf(u, [mf[f"mf_{px}_low_a"],  mf[f"mf_{px}_low_b"],
                                       mf[f"mf_{px}_low_c"],  mf[f"mf_{px}_low_d"]])
        var['med']  = fuzz.trimf (u, [mf[f"mf_{px}_med_a"],  mf[f"mf_{px}_med_b"],
                                       mf[f"mf_{px}_med_c"]])
        var['high'] = fuzz.trapmf(u, [mf[f"mf_{px}_high_a"], mf[f"mf_{px}_high_b"],
                                       mf[f"mf_{px}_high_c"], mf[f"mf_{px}_high_d"]])

    mfs(face,  'face')
    mfs(stab,  'stab')
    mfs(gaze,  'gaze')
    mfs(blink, 'blink')

    conf['very_low']  = fuzz.trapmf(u, [0.00, 0.00, 0.10, 0.20])
    conf['low']       = fuzz.trimf (u, [0.10, 0.25, 0.40])
    conf['medium']    = fuzz.trimf (u, [0.30, 0.50, 0.70])
    conf['high']      = fuzz.trimf (u, [0.60, 0.75, 0.90])
    conf['very_high'] = fuzz.trapmf(u, [0.80, 0.90, 1.00, 1.00])

    lv = {'low': 0, 'med': 1, 'high': 2}

    def out(f, s, g, b):
        pf,ps,pg,pb = lv[f],lv[s],lv[g],lv[b]
        if pg==0 and ps==0:                          return 'very_low'
        if pg==2 and ps==2 and pb>=1 and pf>=1:      return 'very_high'
        t = pg*2 + ps*2 + pf + pb
        if t<=2: return 'very_low'
        if t<=4: return 'low'
        if t<=6: return 'medium'
        if t<=8: return 'high'
        return 'very_high'

    rules = [
        ctrl.Rule(face[f] & stab[s] & gaze[g] & blink[b], conf[out(f,s,g,b)])
        for f in ('low','med','high')
        for s in ('low','med','high')
        for g in ('low','med','high')
        for b in ('low','med','high')
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


# ═══════════════════════════════════════════════════════════════════════════════
# Video Capture  — identical signature to original
# ═══════════════════════════════════════════════════════════════════════════════
def capture_video(
    output_path="temp_video.mp4",
    duration=8,
    fps=30,
    cam_index=0
):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🎥 Recording video... Look at the camera")
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print("🛑 Recording finished")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Video Confidence Analyzer  — identical signature & return keys to original
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_video(video_path):
    p = _PARAMS

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
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
    angle_buf        = _AngleBuffer(size=int(p["angle_smooth_window"]))
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

        # Head stability (nose-X rolling std)
        nose_x = lm[1].x
        nose_x_history.append(nose_x)
        if len(nose_x_history) >= stab_win:
            stability_windows.append(np.std(nose_x_history[-stab_win:]))

        # Gaze via head pose
        pitch, yaw, _ = _head_pose(lm_2d_px, img_h, img_w)
        angle_buf.add([pitch, yaw, 0])
        s_pitch, s_yaw, _ = angle_buf.get_average()
        if (abs(s_yaw) > p["gaze_yaw_threshold"] or
                abs(s_pitch) > p["gaze_pitch_threshold"]):
            gaze_away_frames += 1

        # EAR blink with consecutive-frame guard
        ear = _compute_ear(lm_3d)
        if ear <= p["ear_threshold"]:
            ear_counter += 1
        else:
            if ear_counter >= int(p["ear_consec_frames"]):
                blink_count += 1
            ear_counter = 0

    cap.release()
    landmarker.close()

    if total_frames == 0 or face_frames == 0:
        return {
            "confidence":      0.0,
            "face_ratio":      0.0,
            "head_stability":  0.0,
            "gaze_presence":   0.0,
            "blink_rate":      0.0,
            "blink_score":     0.0,
            "frames_analyzed": int(total_frames)
        }

    # ── Raw metrics ───────────────────────────────────────────────────────────
    face_ratio     = float(np.clip(face_frames / total_frames, 0, 1))
    raw_stab       = np.mean(stability_windows) if stability_windows else 0.0
    head_stability = float(np.clip(np.exp(-raw_stab * p["stability_exp_scale"]), 0, 1))
    gaze_presence  = float(np.clip(1.0 - (gaze_away_frames / face_frames), 0, 1))
    duration_min   = (total_frames / fps) / 60.0
    blink_rate     = min(blink_count / max(duration_min, 1e-6), p["blink_max_reasonable"])
    blink_score    = float(np.clip(
        np.exp(-((blink_rate - p["blink_ideal"]) ** 2) /
               (2 * p["blink_sigma"] ** 2)), 0, 1))

    # ── Fuzzy inference ───────────────────────────────────────────────────────
    try:
        sim = _build_fuzzy_system()
        sim.input['face_ratio']     = face_ratio
        sim.input['head_stability'] = head_stability
        sim.input['gaze_presence']  = gaze_presence
        sim.input['blink_score']    = blink_score
        sim.compute()
        confidence = float(np.clip(sim.output['confidence'], 0, 1))
    except Exception:
        # Fallback: equal-weight average
        confidence = float(np.clip(
            (face_ratio + head_stability + gaze_presence + blink_score) / 4, 0, 1))

    # ── Return — identical keys to original ──────────────────────────────────
    return {
        "confidence":      round(confidence,      3),
        "face_ratio":      round(face_ratio,       3),
        "head_stability":  round(head_stability,   3),
        "gaze_presence":   round(gaze_presence,    3),
        "blink_rate":      round(blink_rate,       2),
        "blink_score":     round(blink_score,      3),
        "frames_analyzed": int(total_frames)
    }