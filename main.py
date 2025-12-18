import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.app_config import load_config
from src.ui import CaptionWindow, ConfigWindow
from src.audio_handler import get_audio_devices

if __name__ == "__main__":
    # Check if config exists to auto-start
    config = load_config()
    
    # If we have a valid config, try to start directly
    if config.get("device_name") and config.get("model_size"):
        # We need to find the device index first
        try:
            devices = get_audio_devices()
            device_index = None
            for i, device in enumerate(devices):
                if device['name'] == config["device_name"]:
                    device_index = i
                    break
            
            if device_index is not None:
                # Direct Start
                print(f"🚀 Auto-starting with saved config: {config}")
                CaptionWindow(
                    config["model_size"],
                    device_index,
                    config["device_name"], 
                    config.get("language", "en"), 
                    config.get("translation_lang", "es")
                )
            else:
                # Device not found, show config
                print("⚠️ Saved device not found. Opening settings...")
                ConfigWindow()
        except Exception as e:
            print(f"❌ Error during auto-start: {e}")
            ConfigWindow()
    else:
        ConfigWindow()

