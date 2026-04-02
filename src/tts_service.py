import asyncio
import pyttsx3

class TTSService:
    async def speak(self, text):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_blocking, text)

    def _speak_blocking(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
