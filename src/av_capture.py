# src/av_capture.py
import threading
from src.audio_recorder import AudioRecorder
from src.video_recorder import VideoRecorder

def record_audio_video():
    stop_event = threading.Event()

    video = VideoRecorder(stop_event=stop_event)
    audio = AudioRecorder(stop_event=stop_event)

    video_thread = threading.Thread(target=video.start)
    video_thread.start()

    # 🔊 Audio blocks until silence
    audio_path = audio.record()

    # 🛑 Audio decides when everything ends
    stop_event.set()
    video_thread.join()

    if audio_path is None:
        return None, None

    return audio_path, video.path
