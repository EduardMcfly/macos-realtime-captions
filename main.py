import os
from app_config import load_config
from ui import CaptionWindow, ConfigWindow
from audio_handler import get_audio_devices

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
                CaptionWindow(config["model_size"], device_index, config["device_name"], config.get("language", "es"))
            else:
                # Device not found, show config
                print("⚠️ Saved device not found. Opening settings...")
                ConfigWindow()
        except Exception as e:
            print(f"❌ Error during auto-start: {e}")
            ConfigWindow()
    else:
        ConfigWindow()

