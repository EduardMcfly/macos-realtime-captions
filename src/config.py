import json
import os
from typing import Dict, Any

# Constants
SAMPLE_RATE = 16000
LOG_FILE = "transcriptions.txt"
CONFIG_FILE = "config.json"

class ConfigManager:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    @staticmethod
    def save_config(
        device_name: str, 
        model_size: str, 
        language: str, 
        translation_lang: str = "en", 
        translation_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    ) -> None:
        config = {
            "device_name": device_name,
            "model_size": model_size,
            "language": language,
            "translation_lang": translation_lang,
            "translation_model": translation_model
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

