import argparse
import os
import sys
import time
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import whisper
import pyttsx3
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            print("[ERROR] OPENROUTER_API_KEY not set in .env")
            sys.exit(1)
        # Use DeepSeek free model via OpenRouter
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.sample_rate = 16000
        self.channels = 1
        self.record_seconds = 8.0


def record_audio(seconds, sample_rate, channels):
    print(f"🎙️ Recording for {seconds:.1f}s... (speak now)")
    sd.default.samplerate = sample_rate
    sd.default.channels = channels
    audio = sd.rec(int(seconds * sample_rate), dtype='int16')
    sd.wait()
    print("✅ Recording complete")
    return audio.reshape(-1, channels)

def save_wav(audio, sample_rate, path):
    wav_write(path, sample_rate, audio)
    return path


def transcribe_audio(path):
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"].strip()


def ask_openrouter(user_text, cfg: Config):
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "You are Jarvis, a concise, helpful AI assistant."},
            {"role": "user", "content": user_text}
        ],
    }
    resp = requests.post(cfg.api_url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"].strip()


engine = pyttsx3.init()
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Jarvis v1 — DeepSeek edition")
    parser.add_argument("--seconds", type=float, default=cfg.record_seconds, help="Record seconds per turn")
    parser.add_argument("--text", action="store_true", help="Use typed input instead of microphone")
    args = parser.parse_args()

    print("\n🤖 Jarvis v1 (DeepSeek) ready. Say 'exit' or 'quit' to stop.\n")

    while True:
        try:
            if args.text:
                user_text = input("You: ").strip()
            else:
                input("Press Enter to record...")
                audio = record_audio(args.seconds, cfg.sample_rate, cfg.channels)
                tmp_wav = os.path.join(os.getcwd(), "jarvis_input.wav")
                save_wav(audio, cfg.sample_rate, tmp_wav)
                user_text = transcribe_audio(tmp_wav)
                print(f"You (transcribed): {user_text}")

            if not user_text:
                print("(Heard nothing)\n")
                continue
            if user_text.lower() in {"exit", "quit"}:
                print("Bye! 👋")
                break

            reply = ask_openrouter(user_text, cfg)
            print(f"Jarvis: {reply}")
            speak_text(reply)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye! 👋")
            break
        except Exception as e:
            print(f"[Error] {type(e).__name__}: {e}")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
