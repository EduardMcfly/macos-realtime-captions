import queue
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import mlx_whisper

# ... (rest of imports)

# ... (Config variables)
SAMPLE_RATE = 16000
BLOCK_SECONDS = 2 # Increased slightly for MLX batching efficiency
LOG_FILE = "transcriptions.txt"

# ... (Audio queue setup)

# ... (CaptionWindow init and UI methods)

    def start_processing(self):
        global audio_buffer
        
        try:
            self.root.after(0, self.set_status, "🎤 Listening (MLX Optimized)...")
            print(f"✅ Starting MLX Whisper ({self.model_size})...")

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

                        # MLX Whisper Transcribe
                        # mlx_whisper expects raw audio or file, we pass numpy array directly
                        result = mlx_whisper.transcribe(
                            chunk.flatten(),
                            path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
                            language=self.language,
                            verbose=False
                        )
                        
                        text = result["text"].strip()

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

        # Language Selection
        ttk.Label(self.root, text="Select Language:").pack(pady=10)
        self.lang_combo = ttk.Combobox(self.root, width=40)
        self.lang_combo['values'] = ["es", "en", "fr", "de", "it", "pt", "auto"]
        self.lang_combo.pack(pady=5)
        self.lang_combo.set("es") # Default

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
        selected_lang = self.lang_combo.get()
        
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
        CaptionWindow(selected_model, device_index, selected_device_name, selected_lang)

if __name__ == "__main__":
    ConfigWindow()
