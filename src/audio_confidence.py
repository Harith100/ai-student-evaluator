import numpy as np
import librosa


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


class AudioConfidenceService:
    def __init__(
        self,
        target_sr=16000,
        silence_threshold=0.01,
        min_pause_duration=0.6,
        hop_length=512
    ):
        self.target_sr = target_sr
        self.silence_threshold = silence_threshold
        self.min_pause_duration = min_pause_duration
        self.hop_length = hop_length

    async def analyze(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.target_sr)

        # ---- Energy (RMS) ----
        rms = librosa.feature.rms(
            y=y,
            hop_length=self.hop_length
        )[0]

        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # ---- Pitch (F0) ----
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            hop_length=self.hop_length
        )

        voiced_f0 = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # ---- Speech ratio ----
        speech_ratio = float(np.mean(rms > self.silence_threshold))

        # ---- PAUSE DETECTION ----
        frame_duration = self.hop_length / sr

        pauses = 0
        silence_start = None

        for i, energy in enumerate(rms):
            t = i * frame_duration

            if energy < self.silence_threshold:
                if silence_start is None:
                    silence_start = t
            else:
                if silence_start is not None:
                    silence_len = t - silence_start
                    if silence_len >= self.min_pause_duration:
                        pauses += 1
                silence_start = None

        # Handle trailing silence
        if silence_start is not None:
            silence_len = (len(rms) * frame_duration) - silence_start
            if silence_len >= self.min_pause_duration:
                pauses += 1

        # ---- Confidence heuristic ----
        confidence = self._confidence_score(
            speech_ratio, pauses, pitch_std
        )

        return {
            "confidence": round(confidence, 3),
            "speech_ratio": round(speech_ratio, 3),
            "pauses": pauses,
            "prosody": {
                "rms_mean": round(rms_mean, 6),
                "rms_std": round(rms_std, 6),
                "pitch_mean_hz": round(pitch_mean, 2),
                "pitch_std_hz": round(pitch_std, 2)
            }
        }

    async def analyze_empty(self):
        return {
            "confidence": 0.0,
            "speech_ratio": 0.0,
            "pauses": 0,
            "prosody": None,
            "reason": "no_speech"
        }

    def _confidence_score(self, speech_ratio, pauses, pitch_std):
        # ---- Fuzzify inputs ----
        sr_low    = _trapmf(speech_ratio, 0.0, 0.0, 0.10, 0.35)
        sr_medium = _trimf(speech_ratio,  0.10, 0.40, 0.70)
        sr_high   = _trapmf(speech_ratio, 0.50, 0.75, 1.0, 1.0)

        ps_flat       = _trapmf(pitch_std, 0.0, 0.0, 10.0, 30.0)
        ps_moderate   = _trimf(pitch_std,  10.0, 45.0, 85.0)
        ps_expressive = _trapmf(pitch_std, 55.0, 90.0, 200.0, 200.0)

        pa_few  = _trapmf(pauses, 0.0, 0.0, 1.0, 4.0)
        pa_some = _trimf(pauses,  1.0, 4.5, 8.0)
        pa_many = _trapmf(pauses, 6.0, 9.0, 99.0, 99.0)

        # ---- Rule base (Mamdani, min-AND / max-OR) ----
        high_act = max(
            min(sr_high,   ps_expressive, pa_few),
            min(sr_high,   ps_expressive, pa_some),
            min(sr_high,   ps_moderate,   pa_few),
            min(sr_medium, ps_expressive, pa_few),
        )

        medium_act = max(
            min(sr_high,   ps_moderate,   pa_some),
            min(sr_medium, ps_moderate,   pa_few),
            min(sr_medium, ps_moderate,   pa_some),
            min(sr_medium, ps_expressive, pa_some),
            min(sr_high,   ps_flat,       pa_few),
            min(sr_medium, ps_flat,       pa_few),
        )

        low_act = max(
            min(sr_low, ps_flat,       pa_many),
            min(sr_low, ps_moderate,   pa_many),
            min(sr_low, ps_expressive, pa_many),
            min(sr_medium, ps_flat,    pa_many),
            min(sr_high,   ps_flat,    pa_many),
            sr_low,
        )

        # ---- Defuzzify (weighted centroid of singleton output sets) ----
        total = high_act + medium_act + low_act
        if total < 1e-9:
            return 0.0

        score = (high_act * 0.85 + medium_act * 0.50 + low_act * 0.15) / total
        return max(0.0, min(1.0, score))