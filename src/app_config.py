import queue
import threading
import json
import os

# Constants
SAMPLE_RATE = 16000
LOG_FILE = "transcriptions.txt"
CONFIG_FILE = "config.json"

# Shared State
audio_queue = queue.Queue()
stop_event = threading.Event()
pause_event = threading.Event() # UI requests pause
transcription_paused = threading.Event() # Transcriber confirms it is paused

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(device_name, model_size, language, translation_lang="en", translation_model="mlx-community/Llama-3.2-1B-Instruct-4bit"):
    config = {
        "device_name": device_name,
        "model_size": model_size,
        "language": language,
        "translation_lang": translation_lang,
        "translation_model": translation_model
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

