import unreal_engine as ue
from gtts import gTTS
import os
from mpg123 import Mpg123
import wave
import sounddevice as sd
import soundfile as sf
import time

MP3_FILE_PATH = os.path.join(ue.get_content_dir(), 'audio.mp3')
WAVE_FILE_PATH = os.path.join(ue.get_content_dir(), 'audio.wav')


class Ue4Speaker:
    def __init__(self):
        self.speaker = Speaker()

    def begin_play(self):
        self.speaker.speak("This is a pen.")


class Speaker:
    def speak(self, text):
        start_at = time.time()
        tts = gTTS(text=text, lang='en')
        tts.save(MP3_FILE_PATH)

        mp3 = Mpg123(MP3_FILE_PATH, library_path=os.path.join(
            ue.get_content_dir(), 'libmpg123-0.dll'))

        rate, channels, encoding = mp3.get_format()

        wav = wave.open(WAVE_FILE_PATH, 'wb')
        wav.setnchannels(channels)
        wav.setframerate(rate)
        wav.setsampwidth(mp3.get_width_by_encoding(encoding))

        for frame in mp3.iter_frames():
            wav.writeframes(frame)

        wav.close()

        data, fs = sf.read(WAVE_FILE_PATH, dtype='float32')
        sd.play(data, fs)
        ue.log("speak: {}".format(time.time() - start_at))
        return
