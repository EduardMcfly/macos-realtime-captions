import queue
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import os
import json
import mlx_whisper

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
BLOCK_SECONDS = 2.0 # Balanced for speed and fluidity
LOG_FILE = "transcriptions.txt"
CONFIG_FILE = "config.json"
# ----------------------------------------

audio_queue = queue.Queue()
# Note: audio_buffer is now local to the processing thread to avoid global state issues
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    if not stop_event.is_set():
        audio_queue.put(indata.copy())

def log_to_file(text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

class CaptionWindow:
    def __init__(self, model_size, device_index, device_name, language):
        print("🔧 Initializing CaptionWindow UI...")
        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"DEBUG: Failed to create tk.Tk(): {e}")
            raise e
        self.root.title(f"Live Captions - {device_name} ({model_size}) [{language}]")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="black")
        self.root.geometry("800x200+200+650")
        
        self.last_text_time = datetime.datetime.now()
        self.paragraph_threshold = 2.0 
        self.language = language if language != "auto" else None

        # UI Components
        config_btn = tk.Button(self.root, text="⚙️", font=("Arial", 14), bg="black", fg="white", bd=0, command=self.open_settings)
        config_btn.place(relx=1.0, x=-10, y=10, anchor="ne")

        self.text_area = tk.Text(self.root, font=("Helvetica", 20), fg="white", bg="black", wrap="word", height=5, bd=0, highlightthickness=0)
        self.text_area.pack(fill="both", expand=True, padx=15, pady=15)
        self.text_area.insert("1.0", "⏳ Loading Model... Please wait.\n")
        self.text_area.mark_set("stable_end", "insert")
        self.text_area.mark_gravity("stable_end", tk.LEFT)
        self.text_area.tag_config("unstable", foreground="#888888")
        self.text_area.config(state="disabled")

        self.model_size = model_size
        self.device_index = device_index
        
        # Start processing in thread but keep reference
        self.processing_thread = threading.Thread(target=self.start_processing, daemon=True)
        self.processing_thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start mainloop explicitly
        print("🖥️ Entering Main Loop...")
        self.root.mainloop()

    def open_settings(self):
        if messagebox.askyesno("Settings", "Change settings? This will restart the captions."):
            stop_event.set()
            try:
                self.root.destroy()
            except:
                pass
            ConfigWindow()

    def on_close(self):
        stop_event.set()
        try:
            self.root.destroy()
        except:
            pass
        # We need to exit the process completely to stop threads
        import os
        os._exit(0)

    def update_text(self, text, is_final=False): 
        try:
            if not self.text_area.winfo_exists(): return
            
            self.text_area.config(state="normal")
            
            # Delete any existing unstable text
            self.text_area.delete("stable_end", tk.END)
            
            if text:
                prefix = ""
                current_time = datetime.datetime.now()
                time_diff = (current_time - self.last_text_time).total_seconds()
                
                if time_diff > self.paragraph_threshold:
                    prefix = "\n\n"
                else:
                    # Check if we need a space
                    prev_char = self.text_area.get("stable_end - 1 chars", "stable_end")
                    if prev_char and prev_char not in [" ", "\n"] and not text.startswith(" "):
                        prefix = " "

                full_text = prefix + text
                
                if is_final:
                    self.text_area.insert("stable_end", full_text)
                    self.text_area.mark_set("stable_end", tk.END) 
                    self.last_text_time = current_time
                else:
                    self.text_area.insert("stable_end", full_text, "unstable")
                
            self.text_area.see(tk.END)
            self.text_area.config(state="disabled")
        except: pass

    def set_status(self, text):
        try:
            if not self.text_area.winfo_exists(): return
            self.text_area.config(state="normal")
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", text + "\n")
            self.text_area.config(state="disabled")
        except: pass

    def start_processing(self):
        # Buffer initialization local to thread
        local_audio_buffer = np.zeros((0, 1), dtype=np.float32)
        last_fast_transcribe_time = 0
        import time

        try:
            self.root.after(0, self.set_status, "...")
            print(f"✅ Starting MLX Whisper ({self.model_size})...")

            # Initialize Whisper Model ONCE here to avoid reloading in loop
            import os
            os.environ["TQDM_DISABLE"] = "1"
            
            # 2. Start Audio Stream
            with sd.InputStream(
                device=self.device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_callback
            ):
                while not stop_event.is_set():
                    # Process audio queue safely
                    while not audio_queue.empty():
                        data = audio_queue.get()
                        local_audio_buffer = np.concatenate([local_audio_buffer, data])

                    # Current accumulated duration
                    current_duration = len(local_audio_buffer) / SAMPLE_RATE
                    now = time.time()

                    # --- SLOW PATH (Commit every ~BLOCK_SECONDS) ---
                    if current_duration >= BLOCK_SECONDS:
                        chunk_samples = int(SAMPLE_RATE * BLOCK_SECONDS)
                        chunk = local_audio_buffer[:chunk_samples]
                        local_audio_buffer = local_audio_buffer[chunk_samples:]

                        try:
                            result = mlx_whisper.transcribe(
                                chunk.flatten(),
                                path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
                                language=self.language,
                                verbose=False,
                                temperature=0.0,
                                condition_on_previous_text=True
                            )
                            text = result["text"].strip()
                            
                            if text.endswith(".") and len(text.split()) < 4:
                                text = text[:-1]

                            if text:
                                print(f"📝 {text}")
                                log_to_file(text)
                                self.root.after(0, self.update_text, text, True)
                        except Exception as e:
                            print(f"\n⚠️ Transcription error: {e}")
                    
                    # --- FAST PATH (Preview every ~0.3s) ---
                    elif current_duration > 0.5 and (now - last_fast_transcribe_time > 0.3):
                        try:
                            result = mlx_whisper.transcribe(
                                local_audio_buffer.flatten(),
                                path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
                                language=self.language,
                                verbose=False,
                                temperature=0.0,
                                condition_on_previous_text=True
                            )
                            text = result["text"].strip()
                            if text:
                                self.root.after(0, self.update_text, text, False)
                            last_fast_transcribe_time = now
                        except: pass
                    
                    sd.sleep(50)

        except Exception as e:
            print(f"Critical Error: {e}")
            pass


class ConfigWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Setup Live Captions")
        self.root.geometry("400x350")
        
        # Load previous config
        self.config = self.load_config()

        # Audio Device Selection
        ttk.Label(self.root, text="Select Audio Device (Microphone/BlackHole):").pack(pady=10)
        self.device_combo = ttk.Combobox(self.root, width=40)
        self.device_combo.pack(pady=5)
        
        self.devices = self.get_audio_devices()
        device_names = [d['name'] for d in self.devices]
        self.device_combo['values'] = device_names
        
        # Auto-select based on saved config or default logic
        saved_device = self.config.get("device_name")
        if saved_device and saved_device in device_names:
             self.device_combo.set(saved_device)
        else:
             # Default search for blackhole
             blackhole_idx = next((i for i, name in enumerate(device_names) if "BlackHole" in name), 0)
             if blackhole_idx < len(device_names):
                self.device_combo.current(blackhole_idx)
             elif device_names:
                self.device_combo.current(0)

        # Model Selection
        ttk.Label(self.root, text="Select Whisper Model Size:").pack(pady=10)
        self.model_combo = ttk.Combobox(self.root, width=40)
        self.model_combo['values'] = ["tiny", "base", "small", "medium", "large-v3"]
        self.model_combo.pack(pady=5)
        self.model_combo.set(self.config.get("model_size", "small"))
        
        ttk.Label(self.root, text="(tiny = fastest/less accurate, medium = slower/more accurate)", font=("Arial", 10), foreground="gray").pack()

        # Language Selection
        ttk.Label(self.root, text="Select Language:").pack(pady=10)
        self.lang_combo = ttk.Combobox(self.root, width=40)
        self.lang_combo['values'] = ["es", "en", "fr", "de", "it", "pt", "auto"]
        self.lang_combo.pack(pady=5)
        self.lang_combo.set(self.config.get("language", "es"))

        # Start Button
        ttk.Button(self.root, text="Start Captions", command=self.start_app).pack(pady=30)
        
        self.root.mainloop()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self, device_name, model_size, language):
        config = {
            "device_name": device_name,
            "model_size": model_size,
            "language": language
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

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

        # Save configuration for next time
        self.save_config(selected_device_name, selected_model, selected_lang)

        # Destroy config window
        try:
             self.root.destroy()
        except:
             pass
             
        # Launch main window
        CaptionWindow(selected_model, device_index, selected_device_name, selected_lang)

if __name__ == "__main__":
    # Check if config exists to auto-start
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            
            # If we have a valid config, try to start directly
            if config.get("device_name") and config.get("model_size"):
                # We need to find the device index first
                try:
                    devices = sd.query_devices()
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
        except Exception as e:
             print(f"❌ Config error: {e}")
             ConfigWindow()
    else:
        ConfigWindow()
