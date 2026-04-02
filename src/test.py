import sounddevice as sd
import numpy as np

fs = 16000
print("Speak...")
audio = sd.rec(int(2 * fs), samplerate=fs, channels=1, device=14)
sd.wait()
print("RMS:", np.sqrt(np.mean(audio**2)))
