import os
import speech_recognition as sr
from gtts import gTTS

class VoiceEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def text_to_speech(self, text: str, output_path: str = "output.mp3"):
        """
        Converts text to speech and saves it to an MP3 file.
        """
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        return output_path

    def speech_to_text(self, audio_file_path: str) -> str:
        """
        Converts an audio file (wav, webm, etc.) to text.
        Tries local speech recognition, with a fallback to Gemini Multimodal Audio API.
        """
        try:
            # Local recognition expects WAV format with standard headers
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            print("VoiceEngine: Local speech recognition succeeded.")
            return text
        except Exception as local_err:
            print(f"VoiceEngine: Local speech recognition failed ({local_err}). Falling back to Gemini Multimodal Audio STT...")
            try:
                from google import genai
                from google.genai import types
                
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key or api_key == "your_api_key_here":
                    raise ValueError("Gemini API key is not configured for fallback STT. Please set GEMINI_API_KEY in .env")
                
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                    
                # Determine mime type based on extension
                ext = os.path.splitext(audio_file_path)[1].lower()
                mime_type = "audio/wav"
                if ext == ".webm":
                    mime_type = "audio/webm"
                elif ext in [".ogg", ".oga", ".spx"]:
                    mime_type = "audio/ogg"
                elif ext == ".mp3":
                    mime_type = "audio/mp3"
                    
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type=mime_type
                        ),
                        "Transcribe the spoken audio text in this file. Output ONLY the transcribed text. If the audio is silent or contains no speech, output an empty string."
                    ]
                )
                
                transcription = response.text.strip() if response.text else ""
                print(f"VoiceEngine: Gemini STT succeeded. Transcript: '{transcription}'")
                return transcription
                
            except Exception as gemini_err:
                print(f"VoiceEngine: Gemini STT fallback failed: {gemini_err}")
                raise Exception(f"Voice Recognition failed. Please check your mic and Gemini API Key. Details: {gemini_err}")
