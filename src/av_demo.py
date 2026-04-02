from src.av_capture import record_audio_video


audio_path, video_path = record_audio_video()

if audio_path:
    print("✅ Audio saved:", audio_path)
    print("✅ Video saved:", video_path)
else:
    print("⚠️ No valid speech detected")
