# utils/audio_processor.py
import pyaudio
import wave
import librosa
import noisereduce as nr
import whisper
import speech_recognition as sr
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import numpy as np
import os
import json
import tempfile
from openai import OpenAI
import pyttsx3
import time

class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=15, use_api=False, api_key=None):
        """
        Initialize audio processor for English audio interaction.
        
        Args:
            sample_rate: Audio sample rate
            duration: Max recording duration in seconds
            use_api: Use OpenAI API for STT and mood analysis
            api_key: OpenAI API key
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_api = use_api
        self.recognizer = sr.Recognizer()
        
        # TTS engine for therapist (English by default)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)    # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
        if use_api:
            if not api_key:
                raise ValueError("API key required for API mode")
            self.openai_client = OpenAI(api_key=api_key)
            self.whisper_model = None
            print("[AUDIO] Using OpenAI API for STT + Mood Analysis (English)")
        else:
            print("[AUDIO] Loading Whisper model (offline, English)...")
            self.whisper_model = whisper.load_model("base")
            self.openai_client = None
            print("[AUDIO] Offline mode ready (English)")

    def speak_text(self, text):
        """TTS: Speak text (for therapist questions)."""
        print(f"🎤 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def record_from_mic(self):
        """Record audio from microphone."""
        print(f"🎙️  Recording... Speak for up to {self.duration} seconds (stops on silence).")
        print("   Say 'stop', 'done', or stay silent to finish.\n")
        
        with sr.Microphone(sample_rate=self.sample_rate) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=self.duration, phrase_time_limit=self.duration)
                print("✅ Recording completed.")
                return audio
            except sr.WaitTimeoutError:
                print("⚠️  No speech detected, treating as 'No response'.")
                return None
            except Exception as e:
                print(f"❌ Recording error: {e}")
                return None

    def stt(self, audio_data):
        """Speech-to-Text: Audio → text (English)."""
        if not audio_data:
            return "No audio input detected."
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data.get_wav_data())
            temp_file.close()
            
        try:
            if self.use_api and self.openai_client:
                with open(temp_path, "rb") as f:
                    result = self.openai_client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=f, 
                        response_format="text",
                        language="en"  # English
                    )
            else:
                # Offline Whisper with English
                result = self.whisper_model.transcribe(temp_path, language="en")["text"]
        except Exception as e:
            print(f"[STT] Error: {e}")
            result = "Speech recognition failed."
        finally:
            os.unlink(temp_path)
        return result.strip()

    def analyze_mood(self, audio_path):
        """Analyze mood from audio file (for PPD screening)."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(y) < 1000:  # Audio too short
                return {"pitch": 0, "tempo": 0, "energy": 0, "sadness_prob": 0.5, "anxiety_prob": 0.3, "energy_level": "low"}
            
            # Noise reduction
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            
            # Extract audio features
            pitch = librosa.yin(y_denoised, fmin=50, fmax=500)
            pitch_mean = np.mean(pitch[pitch > 0]) if len(pitch[pitch > 0]) > 0 else 200
            
            tempo, _ = librosa.beat.beat_track(y=y_denoised, sr=sr)
            energy = librosa.feature.rms(y=y_denoised).mean()
            
            # Mood analysis for PPD (focus on sadness, anxiety, low energy)
            sadness_score = 0
            anxiety_score = 0
            if pitch_mean < 180:  # Low pitch = sadness
                sadness_score += 0.4
            if tempo < 80:  # Slow tempo = sadness
                sadness_score += 0.3
            if energy < 0.05:  # Low energy = depression
                sadness_score += 0.3
            if tempo > 120 or np.std(pitch) > 50:  # Fast + pitch variation = anxiety
                anxiety_score += 0.6
                
            # Use OpenAI if API available (more accurate)
            if self.use_api:
                try:
                    with open(audio_path, "rb") as f:
                        prompt = """
                        Analyze this audio for postpartum depression (PPD) screening.
                        Focus on: sadness, anxiety, energy level, speech clarity.
                        Return JSON only: {"sadness_prob": 0-1, "anxiety_prob": 0-1, "energy_level": "low|normal|high", "confidence": 0-1}
                        """
                        response = self.openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=100
                        )
                        mood_text = response.choices[0].message.content.strip()
                        # Try to parse JSON response
                        try:
                            mood = json.loads(mood_text)
                        except json.JSONDecodeError:
                            # Fallback if not valid JSON
                            mood = {
                                "sadness_prob": min(1.0, max(0.0, sadness_score)),
                                "anxiety_prob": min(1.0, max(0.0, anxiety_score)),
                                "energy_level": "low" if energy < 0.03 else "normal" if energy < 0.08 else "high",
                                "confidence": 0.7
                            }
                except Exception as e:
                    print(f"[MOOD] OpenAI analysis error: {e}")
                    mood = {
                        "sadness_prob": min(1.0, max(0.0, sadness_score)),
                        "anxiety_prob": min(1.0, max(0.0, anxiety_score)),
                        "energy_level": "low" if energy < 0.03 else "normal" if energy < 0.08 else "high",
                        "confidence": 0.7
                    }
            else:
                # Offline mood estimation
                mood = {
                    "sadness_prob": min(1.0, max(0.0, sadness_score)),
                    "anxiety_prob": min(1.0, max(0.0, anxiety_score)),
                    "energy_level": "low" if energy < 0.03 else "normal" if energy < 0.08 else "high",
                    "confidence": 0.7
                }
                
            return {
                "pitch": float(pitch_mean),
                "tempo": float(tempo),
                "energy": float(energy),
                **mood
            }
        except Exception as e:
            print(f"[MOOD] Analysis error: {e}")
            return {"pitch": 0, "tempo": 0, "energy": 0, "sadness_prob": 0.5, "anxiety_prob": 0.3, "energy_level": "normal", "confidence": 0.5}

    def process_response(self, audio_data):
        """Full pipeline: Record → STT + Mood Analysis."""
        if not audio_data:
            return {"text": "No audio input", "mood": {"sadness_prob": 0.6, "energy_level": "low"}}
        
        # Create temp file for mood analysis
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data.get_wav_data())
            temp_file.close()
            
        try:
            text = self.stt(audio_data)
            mood = self.analyze_mood(temp_path)
        except Exception as e:
            print(f"[AUDIO] Processing error: {e}")
            text = "Audio processing failed"
            mood = {"sadness_prob": 0.5, "energy_level": "normal"}
        finally:
            os.unlink(temp_path)
            
        print(f"Transcribed: '{text}'")
        print(f"Mood Analysis: Sadness={mood.get('sadness_prob', 0):.1%}, Anxiety={mood.get('anxiety_prob', 0):.1%}, Energy={mood.get('energy_level', 'normal')}")
        
        return {"text": text, "mood": mood}