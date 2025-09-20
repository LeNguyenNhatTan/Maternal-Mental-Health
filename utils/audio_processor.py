import pyaudio
import wave
import librosa
import noisereduce as nr
import whisper
import speech_recognition as sr
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import numpy as np
import os
from openai import OpenAI  

class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=15, use_api=False, api_key=None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_api = use_api
        self.recognizer = sr.Recognizer()
        if use_api:
            if not api_key:
                raise ValueError("API key required for API mode")
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.whisper_model = whisper.load_model("base")  # Offline STT

    def record_from_mic(self):
        """Record from mic."""
        print("Recording... Speak ({}s).".format(self.duration))
        with sr.Microphone(sample_rate=self.sample_rate) as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=self.duration, phrase_time_limit=self.duration)
        return audio

    def stt(self, audio_data):
        """STT: Audio → text."""
        audio_file = "temp.wav"
        with open(audio_file, "wb") as f:
            f.write(audio_data.get_wav_data())
        if self.use_api:
            with open(audio_file, "rb") as f:
                result = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text"
                )
        else:
            result = self.whisper_model.transcribe(audio_file)["text"]
        os.remove(audio_file)
        return result

    def analyze_mood(self, audio_file):
        """SER: Analyze mood/prosody."""
        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # Prosody
        pitch = librosa.yin(y_denoised, fmin=50, fmax=500).mean()
        tempo = librosa.beat.beat_track(y=y_denoised, sr=sr)[0]
        energy = librosa.feature.rms(y=y_denoised).mean()

        # SER: Offline pyAudioAnalysis hoặc API OpenAI (GPT-4o analyze)
        if self.use_api:
            with open(audio_file, "rb") as f:
                prompt = "Analyze emotion: sadness, energy level. Output JSON: {'sadness_prob': 0.7, ...}"
                result = self.openai_client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": prompt}], files=[f]
                )
            mood = json.loads(result.choices[0].message.content)
        else:
            features, _ = ShortTermFeatures.feature_extraction(y_denoised, sr, 0.05 * sr, 0.025 * sr)
            sadness_prob = np.mean(features[0])  # Adjust for sadness
            mood = {"sadness_prob": sadness_prob}

        return {"pitch": pitch, "tempo": tempo, "energy": energy, **mood}

    def process_response(self, audio_data):
        """Full: STT + SER."""
        audio_file = "temp.wav"
        with open(audio_file, "wb") as f:
            f.write(audio_data.get_wav_data())
        text = self.stt(audio_data)
        mood = self.analyze_mood(audio_file)
        os.remove(audio_file)
        return {"text": text, "mood": mood}