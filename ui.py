import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import os
import sounddevice as sd

from app_config import load_config, save_config, stop_event, CONFIG_FILE
from audio_handler import get_audio_devices
from transcriber import run_transcription_loop

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
        self.text_area.config(state="disabled")

        self.model_size = model_size
        self.device_index = device_index
        
        # Start processing in thread but keep reference
        self.processing_thread = threading.Thread(
            target=lambda: run_transcription_loop(
                self.device_index, 
                self.model_size, 
                self.language, 
                self.schedule_update_text,
                self.schedule_set_status
            ), 
            daemon=True
        )
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

    def schedule_update_text(self, text, is_final=True):
        self.root.after(0, self.update_text, text, is_final)

    def schedule_set_status(self, text):
        self.root.after(0, self.set_status, text)

    def update_text(self, text, is_final=True): 
        try:
            if not self.text_area.winfo_exists(): return
            
            self.text_area.config(state="normal")
            
            # Check scroll position before modifying text
            # yview() returns (top, bottom) as fractions between 0 and 1
            # If bottom is near 1.0, we are at the bottom and should auto-scroll
            was_at_bottom = self.text_area.yview()[1] > 0.99

            # Remove previous pending text safely
            try:
                # We use tag indices provided by tkinter when a tag exists
                self.text_area.delete("pending.first", "pending.last")
            except tk.TclError:
                pass # Tag "pending" not present or empty
            
            if text:
                prefix = ""
                # Only add prefix (newlines/spaces) if we are appending permanently or starting a preview
                # Logic: We want the preview to look like it will be committed.
                
                # Check previous character (ignoring the just-deleted pending text)
                prev_char = self.text_area.get("end-2c", "end-1c")
                
                current_time = datetime.datetime.now()
                time_diff = (current_time - self.last_text_time).total_seconds()
                
                if time_diff > self.paragraph_threshold:
                    prefix = "\n\n"
                elif prev_char and prev_char not in [" ", "\n"] and not text.startswith(" "):
                    prefix = " "

                full_text = prefix + text
                
                if is_final:
                    self.text_area.insert(tk.END, full_text)
                    self.last_text_time = current_time
                else:
                    self.text_area.insert(tk.END, full_text, "pending")
                    self.text_area.tag_config("pending", foreground="gray")
                
            if was_at_bottom:
                self.text_area.see(tk.END)
            
            self.text_area.config(state="disabled")
        except Exception as e:
            print(f"Error updating text: {e}")

    def set_status(self, text):
        try:
            if not self.text_area.winfo_exists(): return
            self.text_area.config(state="normal")
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", text + "\n")
            self.text_area.config(state="disabled")
        except: pass


class ConfigWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Setup Live Captions")
        self.root.geometry("400x350")
        
        # Load previous config
        self.config = load_config()

        # Audio Device Selection
        ttk.Label(self.root, text="Select Audio Device (Microphone/BlackHole):").pack(pady=10)
        self.device_combo = ttk.Combobox(self.root, width=40)
        self.device_combo.pack(pady=5)
        
        self.devices = get_audio_devices()
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
        save_config(selected_device_name, selected_model, selected_lang)

        # Destroy config window
        try:
             self.root.destroy()
        except:
             pass
             
        # Launch main window
        CaptionWindow(selected_model, device_index, selected_device_name, selected_lang)

