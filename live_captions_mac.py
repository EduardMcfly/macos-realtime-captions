import queue
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import mlx_whisper

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.5 # Reduced for near real-time streaming feel
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
    def __init__(self, model_size, device_index, device_name, language):
        self.root = tk.Tk()
        self.root.title(f"Live Captions - {device_name} ({model_size}) [{language}]")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="black")
        self.root.geometry("800x200+200+650")
        
        self.last_text_time = datetime.datetime.now()
        self.paragraph_threshold = 2.0 # Seconds of silence to trigger new paragraph
        
        self.language = language if language != "auto" else None

        # Text Area

    def update_text(self, text):
        current_time = datetime.datetime.now()
        time_diff = (current_time - self.last_text_time).total_seconds()
        
        self.text_area.config(state="normal")
        
        # Check if enough time has passed to start a new paragraph
        if time_diff > self.paragraph_threshold:
            self.text_area.insert(tk.END, "\n\n" + text + " ")
        else:
            self.text_area.insert(tk.END, text + " ")
            
        self.last_text_time = current_time
        
        # Keep last 15 lines (increased slightly to accommodate paragraphs)
        # ... (lógica de limpieza)
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
        current_time = datetime.datetime.now()
        time_diff = (current_time - self.last_text_time).total_seconds()
        
        self.text_area.config(state="normal")
        
        # Check if enough time has passed to start a new paragraph
        if time_diff > self.paragraph_threshold:
            self.text_area.insert(tk.END, "\n\n" + text)
        else:
            # If text doesn't start with space, add one for continuity
            prefix = " " if not text.startswith(" ") else ""
            self.text_area.insert(tk.END, prefix + text)
            
        self.last_text_time = current_time
        
        self.text_area.see(tk.END)
        self.text_area.config(state="disabled")

    def set_status(self, text):
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert("1.0", text + "\n")
        self.text_area.config(state="disabled")

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

                    if len(audio_buffer) >= int(SAMPLE_RATE * BLOCK_SECONDS):
                        chunk = audio_buffer[:int(SAMPLE_RATE * BLOCK_SECONDS)]
                        audio_buffer = audio_buffer[int(SAMPLE_RATE * BLOCK_SECONDS):]

                        # MLX Whisper Transcribe
                        try:
                            # Use step-by-step decoding if available or smaller chunks
                            import sys
                            import os
                            
                            # Suppress TQDM loading bars from mlx/huggingface
                            os.environ["TQDM_DISABLE"] = "1"
                            
                            result = mlx_whisper.transcribe(
                                chunk.flatten(),
                                path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
                                language=self.language,
                                verbose=False,
                                # Optimize for speed
                                temperature=0.0,
                                compression_ratio_threshold=None,
                                logprob_threshold=None,
                                no_speech_threshold=None,
                                condition_on_previous_text=False
                            )
                            text = result["text"].strip()

                            if text:
                                print(f"📝 {text}")
                                log_to_file(text)
                                self.root.after(0, self.update_text, text)
                        except Exception as e:
                            print(f"⚠️ Transcription error: {e}")
                    
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
