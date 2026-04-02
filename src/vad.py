import torch
from silero_vad import load_silero_vad, get_speech_timestamps

class VAD:
    def __init__(self, sample_rate=16000):
        self.model = load_silero_vad()
        self.sr = sample_rate

    def has_speech(self, audio):
        # audio: numpy array (float32)
        audio_tensor = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sr,
            min_speech_duration_ms=200,
            min_silence_duration_ms=300
        )
        return len(timestamps) > 0
