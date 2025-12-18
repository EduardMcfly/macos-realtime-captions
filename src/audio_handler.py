import sounddevice as sd
from tkinter import messagebox
from .app_config import audio_queue, stop_event

def audio_callback(indata, frames, time, status):
    if not stop_event.is_set():
        audio_queue.put(indata.copy())

def get_audio_devices():
    try:
        return sd.query_devices()
    except Exception as e:
        messagebox.showerror("Error", f"Could not list audio devices: {e}")
        return []

