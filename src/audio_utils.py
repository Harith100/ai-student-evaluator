import soundfile as sf
import numpy as np

def validate_audio(path, min_seconds=1.0, min_rms=0.01):
    audio, fs = sf.read(path)
    duration = len(audio) / fs
    rms = np.sqrt(np.mean(audio**2))

    return {
        "valid": duration >= min_seconds and rms >= min_rms,
        "duration": duration,
        "rms": rms
    }
