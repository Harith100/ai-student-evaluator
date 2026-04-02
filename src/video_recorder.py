# src/video_recorder.py
import cv2
import tempfile
import os

class VideoRecorder:
    def __init__(self, stop_event, fps=20, cam_index=0):
        self.stop_event = stop_event
        self.fps = fps
        self.cam_index = cam_index
        self.path = None

    def start(self):
        print("🎥 Video recording started")

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        # Check if camera opened successfully
        if not cap.isOpened():
            raise RuntimeError("❌ Camera failed to open")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("❌ Camera opened but no frames received")


        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fd, self.path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        writer = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height)
        )

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()
        print("🛑 Video recording stopped")
