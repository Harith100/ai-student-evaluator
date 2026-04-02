import asyncio
from src.pdf_loader import load_pdf
from src.chunker import chunk_text
from src.adaptive_examiner import AdaptiveExaminer
from src.tts_service import TTSService
from src.stt_service import STTService

from src.av_capture import record_audio_video
from src.audio_confidence import AudioConfidenceService
from src.video_confidence import analyze_video


async def run():
    text = load_pdf("src/data/pdfs/environment.pdf")
    chunks = chunk_text(text)

    examiner = AdaptiveExaminer(chunks)
    tts = TTSService()
    stt = STTService()

    audio_conf = AudioConfidenceService()
    video_conf = analyze_video

    confidence_tasks = []
    confidence_results = []

    print("🎓 Cognitive Examiner Started")

    q_index = 1

    while not examiner.state.done():
        q = await examiner.ask_next()
        print(f"\nQ{q_index}: {q['question']}")
        await tts.speak(q["question"])

        # 🎥🎤 RECORD AUDIO + VIDEO TOGETHER
        audio_path, video_path = record_audio_video()

        if audio_path is None:
            print("⚠️ No clear speech detected.")
            await tts.speak("I didn’t catch that.")
            continue

        # 🔊 STT
        student = await stt.transcribe(audio_path)
        print("You said:", student)

        # 🧠 SEMANTIC GRADING
        result = await examiner.grade(q, student)
        print("Score:", round(result["score"] * 10, 2))
        print("Feedback:", result["feedback"])

        # 🎧🎥 CONFIDENCE — RUN IN PARALLEL
        audio_task = asyncio.create_task(
            audio_conf.analyze(audio_path)
        )

        video_task = asyncio.create_task(
            asyncio.to_thread(analyze_video, video_path)
        )

        confidence_tasks.append((audio_task, video_task))


        q_index += 1

    print("\n🏁 Exam Finished — Confidence Analysis Running...")

    confidence_results = []

    for audio_task, video_task in confidence_tasks:
        audio_res, video_res = await asyncio.gather(audio_task, video_task)
        confidence_results.append((audio_res, video_res))

    print("\n📊 Confidence — Stats for Nerds")
    for i, (audio, video) in enumerate(confidence_results, start=1):
        print(f"\nQ{i}")
        print(f"  AUDIO confidence   : {audio['confidence']}")
        print(f"  speech_ratio       : {audio.get('speech_ratio')}")
        print(f"  pauses             : {audio.get('pauses')}")
        prosody = audio.get("prosody") or {}

        if prosody:
            print(f"  rms_mean           : {prosody.get('rms_mean')}")
        if prosody:
            print(f"  rms_std            : {prosody.get('rms_std')}")
        if prosody:
            print(f"  pitch_meanHz       : {prosody.get('pitch_mean_hz')}")
        if prosody:
            print(f"  pitch_stdHz        : {prosody.get('pitch_std_hz')}")


        print(f"  VIDEO confidence   : {video['confidence']}")
        print(f"  face_ratio         : {video['face_ratio']}")
        print(f"  head_stability     : {video['head_stability']}")
        print(f"  gaze_presence      : {video['gaze_presence']}")
        print(f"  blink_rate         : {video['blink_rate']}")

asyncio.run(run())
