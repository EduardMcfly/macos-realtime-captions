import queue
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
from faster_whisper import WhisperModel, download_model

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
BLOCK_SECONDS = 1   # lower = less delay (1s is faster)
LOG_FILE = "transcriptions.txt"
# ----------------------------------------

audio_queue = queue.Queue()
audio_buffer = np.zeros((0, 1), dtype=np.float32)
stop_event = threading.Event()
model = None

def audio_callback(indata, frames, time, status):
    if not stop_event.is_set():
        audio_queue.put(indata.copy())

def log_to_file(text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

class CaptionWindow:
    def __init__(self, model_size, device_index, device_name):
        self.root = tk.Tk()
        self.root.title(f"Live Captions - {device_name} ({model_size})")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="black")
        self.root.geometry("800x200+200+650")

        # Text Area
        self.text_area = tk.Text(
            self.root,
            font=("Helvetica", 20),
            fg="white",
            bg="black",
            wrap="word",
            height=5,
            bd=0,
            highlightthickness=0
        )
        self.text_area.pack(fill="both", expand=True, padx=15, pady=15)
        self.text_area.insert("1.0", "⏳ Loading Model... Please wait.\n")
        self.text_area.config(state="disabled")

        self.model_size = model_size
        self.device_index = device_index
        
        # Start processing in background
        threading.Thread(target=self.start_processing, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        stop_event.set()
        self.root.destroy()
        # Force exit to kill threads
        import os
        os._exit(0)

    def update_text(self, text):
        self.text_area.config(state="normal")
        self.text_area.insert(tk.END, text + "\n")
        
        # Keep last 10 lines
        lines = self.text_area.get("1.0", tk.END).split("\n")
        if len(lines) > 11:
            self.text_area.delete("1.0", "2.0")
            
        self.text_area.see(tk.END)
        self.text_area.config(state="disabled")

    def set_status(self, text):
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert("1.0", text + "\n")
        self.text_area.config(state="disabled")

    def start_processing(self):
        global model, audio_buffer
        
        try:
            # 1. Load Model
            print(f"⏳ Downloading/Loading model '{self.model_size}'...", flush=True)
            model_path = download_model(self.model_size)
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
            
            self.root.after(0, self.set_status, "🎤 Listening...")
            print("✅ Model loaded. Starting audio stream...")

            # 2. Start Audio Stream
            with sd.InputStream(
                device=self.device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_callback
            ):
                while not stop_event.is_set():
                    # Process audio queue
                    while not audio_queue.empty():
                        audio_buffer = np.concatenate([audio_buffer, audio_queue.get()])

                    if len(audio_buffer) >= SAMPLE_RATE * BLOCK_SECONDS:
                        chunk = audio_buffer[:SAMPLE_RATE * BLOCK_SECONDS]
                        audio_buffer = audio_buffer[SAMPLE_RATE * BLOCK_SECONDS:]

                        segments, _ = model.transcribe(
                            chunk.flatten(),
                            language="es",
                            vad_filter=True,
                            beam_size=1
                        )

                        text = " ".join(s.text for s in segments).strip()
                        if text:
                            print(f"📝 {text}")
                            log_to_file(text)
                            self.root.after(0, self.update_text, text)
                    
                    sd.sleep(50) # Small sleep to reduce CPU

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", str(e))
            self.root.after(0, self.root.destroy)


class ConfigWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Setup Live Captions")
        self.root.geometry("400x350")
        
        # Audio Device Selection
        ttk.Label(self.root, text="Select Audio Device (Microphone/BlackHole):").pack(pady=10)
        self.device_combo = ttk.Combobox(self.root, width=40)
        self.device_combo.pack(pady=5)
        
        self.devices = self.get_audio_devices()
        device_names = [d['name'] for d in self.devices]
        self.device_combo['values'] = device_names
        
        # Auto-select BlackHole if found, otherwise first device
        blackhole_idx = next((i for i, name in enumerate(device_names) if "BlackHole" in name), 0)
        self.device_combo.current(blackhole_idx)

        # Model Selection
        ttk.Label(self.root, text="Select Whisper Model Size:").pack(pady=10)
        self.model_combo = ttk.Combobox(self.root, width=40)
        self.model_combo['values'] = ["tiny", "base", "small", "medium", "large-v3"]
        self.model_combo.pack(pady=5)
        self.model_combo.set("small") # Default
        
        ttk.Label(self.root, text="(tiny = fastest/less accurate, medium = slower/more accurate)", font=("Arial", 10), foreground="gray").pack()

        # Start Button
        ttk.Button(self.root, text="Start Captions", command=self.start_app).pack(pady=30)
        
        self.root.mainloop()

    def get_audio_devices(self):
        try:
            return sd.query_devices()
        except Exception as e:
            messagebox.showerror("Error", f"Could not list audio devices: {e}")
            return []

    def start_app(self):
        selected_device_name = self.device_combo.get()
        selected_model = self.model_combo.get()
        
        if not selected_device_name:
            messagebox.showwarning("Warning", "Please select an audio device.")
            return

        # Find device index
        device_index = None
        for i, device in enumerate(self.devices):
            if device['name'] == selected_device_name:
                device_index = i
                break
        
        if device_index is None:
            messagebox.showerror("Error", "Selected device not found.")
            return

        self.root.destroy()
        # Launch main window
        CaptionWindow(selected_model, device_index, selected_device_name)

if __name__ == "__main__":
    ConfigWindow()
