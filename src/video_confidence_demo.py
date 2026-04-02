# src/video_confidence_demo.py
from src.video_confidence import capture_video, analyze_video

video = "temp_video.mp4"
stats = analyze_video(video)

print("\n🎥 Video Confidence — Stats for Nerds\n")
for k, v in stats.items():
    print(f"{k:16}: {v}")
