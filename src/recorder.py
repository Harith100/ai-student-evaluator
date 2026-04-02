import pyaudio
import numpy as np
import wave
import tempfile
import os
import time


class Recorder:
    def __init__(
        self,
        silence_threshold=80,      # RMS threshold for int16 , 
        silence_duration=1.2,       # seconds of silence to stop
        min_speech_duration=0.8,    # minimum speech length (IMPORTANT)
        max_duration=20.0,          # hard safety cap
        chunk_ms=30                 # chunk size in ms
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_duration = max_duration
        self.chunk_ms = chunk_ms

        self.fs = 16000
        self.channels = 1
        self.format = pyaudio.paInt16

        self.p = pyaudio.PyAudio()

    def _rms(self, frame: bytes) -> int:
        samples = np.frombuffer(frame, dtype=np.int16)
        if len(samples) == 0:
            return 0
        return int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    async def capture(self) -> str:
        print("🎤 Listening...")

        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.fs,
            input=True,
            frames_per_buffer=int(self.fs * self.chunk_ms / 1000),
        )

        frames = []
        silence_start = None
        speech_detected = False
        speech_start_time = None
        start_time = time.time()

        while True:
            frame = stream.read(
                int(self.fs * self.chunk_ms / 1000),
                exception_on_overflow=False
            )
            frames.append(frame)

            rms = self._rms(frame)
            print(f"RMS: {rms:04d}", end="\r")

            now = time.time()

            if rms > self.silence_threshold:
                if not speech_detected:
                    speech_detected = True
                    speech_start_time = now
                silence_start = None
            else:
                if speech_detected:
                    if silence_start is None:
                        silence_start = now
                    elif (
                        now - silence_start >= self.silence_duration
                        and now - speech_start_time >= self.min_speech_duration
                    ):
                        break

            if now - start_time >= self.max_duration:
                break

        stream.stop_stream()
        stream.close()

        print("\n🛑 Recording stopped")

        if not speech_detected:
            return None

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        wf = wave.open(path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.fs)
        wf.writeframes(b"".join(frames))
        wf.close()

        return path
