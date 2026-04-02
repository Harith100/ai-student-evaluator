import asyncio
from faster_whisper import WhisperModel

class STTService:
    def __init__(self):
        self.model = WhisperModel("small", device="cpu", compute_type="int8")

    async def transcribe(self, audio_path):
        segments, _ = await asyncio.to_thread(self.model.transcribe, audio_path)
        return " ".join(seg.text for seg in segments)
