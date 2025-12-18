import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import os
import sounddevice as sd
from screeninfo import get_monitors

from .app_config import load_config, save_config, stop_event, CONFIG_FILE
from .audio_handler import get_audio_devices
from .transcriber import run_transcription_loop

class CaptionWindow:
    def __init__(self, model_size, device_index, device_name, language, translation_lang="en"):
        print("🔧 Initializing CaptionWindow UI...")
        try:
            self.root = tk.Tk()
        except Exception as e:
            print(f"DEBUG: Failed to create tk.Tk(): {e}")
            raise e
        self.root.title(f"Live Captions - {device_name} ({model_size}) [{language} -> {translation_lang}]")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="black")
        
        # Translation State
        self.translation_lang = translation_lang
        self.translation_model = None
        self.translation_tokenizer = None
        self.is_translating = False
        
        # Window Dimensions
        window_width = 800
        window_height = 200
        padding_bottom = 50

        # Calculate Position (Bottom Center of Monitor with Mouse)
        try:
            # 1. Get Mouse Position
            mouse_x, mouse_y = self.root.winfo_pointerxy()
            
            # 2. Find Monitor containing Mouse
            target_monitor = None
            monitors = get_monitors()
            
            for m in monitors:
                # Check if mouse is within monitor bounds
                # screeninfo m: x, y, width, height
                if (m.x <= mouse_x < m.x + m.width) and (m.y <= mouse_y < m.y + m.height):
                    target_monitor = m
                    break
            
            # Fallback to primary if not found
            if not target_monitor and len(monitors) > 0:
                target_monitor = monitors[0]
                
            if target_monitor:
                # 3. Calculate Center Bottom relative to that monitor
                # x_pos = monitor_x + (monitor_width - window_width) / 2
                x_pos = target_monitor.x + (target_monitor.width - window_width) // 2
                
                # y_pos = monitor_y + monitor_height - window_height - padding
                y_pos = target_monitor.y + target_monitor.height - window_height - padding_bottom
                
                # Ensure it doesn't go off-screen (basic clamp)
                if x_pos < target_monitor.x: x_pos = target_monitor.x
                
                print(f"🖥️ Detected Monitor: {target_monitor.name} ({target_monitor.width}x{target_monitor.height}) at {target_monitor.x},{target_monitor.y}")
                print(f"📍 Placing window at: {x_pos},{y_pos}")
                
                self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
            else:
                # Fallback to standard method if screeninfo fails completely
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                x_pos = (screen_width - window_width) // 2
                y_pos = screen_height - window_height - padding_bottom
                self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        except Exception as e:
            print(f"⚠️ Error calculating monitor position: {e}")
            # Fallback
            self.root.geometry(f"{window_width}x{window_height}+200+600")
        
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
        
        # Click to Translate binding
        self.text_area.bind("<Button-1>", self.on_text_click)

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
        
        # Start translation model loader in background
        threading.Thread(target=self.preload_translation_model, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start mainloop explicitly
        print("🖥️ Entering Main Loop...")
        self.root.mainloop()

    def preload_translation_model(self):
        """Preload the lightweight LLM for translation to avoid lag on first click."""
        try:
            print("🧠 Pre-loading Translation Model (Qwen2.5-0.5B)...")
            from mlx_lm import load
            # Using Qwen2.5-0.5B-Instruct-4bit for extreme speed and low memory
            self.translation_model, self.translation_tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
            print("🧠 Translation Model Ready!")
        except Exception as e:
            print(f"⚠️ Failed to load translation model: {e}")

    def on_text_click(self, event):
        """Handle click to translate current text."""
        if self.is_translating: return
        
        # Visual feedback
        self.root.configure(bg="#1a1a1a") # Slightly lighter bg
        self.text_area.configure(bg="#1a1a1a")
        self.root.after(200, lambda: self.root.configure(bg="black"))
        self.root.after(200, lambda: self.text_area.configure(bg="black"))
        
        threading.Thread(target=self.perform_translation, daemon=True).start()

    def perform_translation(self):
        self.is_translating = True
        try:
            # Get current text
            full_text = self.text_area.get("1.0", tk.END).strip()
            if not full_text or "Loading" in full_text: 
                self.is_translating = False
                return

            print(f"🔄 Translating to {self.translation_lang}...")
            self.schedule_set_status(f"🔄 Translating to {self.translation_lang}...")

            if not self.translation_model:
                self.preload_translation_model()

            from mlx_lm import generate
            
            # Construct Prompt
            messages = [
                {"role": "system", "content": f"You are a professional translator. Translate the following text strictly into {self.translation_lang}. Do not add explanations, just the translation."},
                {"role": "user", "content": full_text}
            ]
            
            prompt = self.translation_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate
            translation = generate(
                self.translation_model, 
                self.translation_tokenizer, 
                prompt=prompt, 
                max_tokens=500, 
                verbose=False
            )
            
            # Update UI with translation
            self.schedule_replace_text(translation.strip())

        except Exception as e:
            print(f"Translation Error: {e}")
            self.schedule_set_status(f"❌ Translation Error: {e}")
            # Revert status after 2s
            self.root.after(2000, lambda: self.schedule_update_text(full_text, True))
        
        self.is_translating = False

    def schedule_replace_text(self, text):
        self.root.after(0, self.replace_text, text)

    def replace_text(self, text):
        try:
            self.text_area.config(state="normal")
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", text)
            self.text_area.config(state="disabled")
        except: pass

    def open_settings(self):
        # Stop processing but keep window open
        stop_event.set()
        
        # Open config window with restart callback
        ConfigWindow(restart_callback=self.restart_processing)

    def restart_processing(self, model_size, device_index, device_name, language, translation_lang):
        # 1. Update internal state
        self.model_size = model_size
        self.device_index = device_index
        self.language = language if language != "auto" else None
        self.translation_lang = translation_lang
        
        # 2. Update Window Title
        self.root.title(f"Live Captions - {device_name} ({model_size}) [{language} -> {translation_lang}]")
        
        # 3. Reset Stop Event
        stop_event.clear()
        
        # 4. Restart Thread
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
        
        self.set_status(f"🔄 Restarted with {model_size}...")

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
    def __init__(self, restart_callback=None):
        self.restart_callback = restart_callback
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
        
        ttk.Label(self.root, text="(Fast 'tiny' model always used for previews)", font=("Arial", 10), foreground="gray").pack()

        # Language Selection
        ttk.Label(self.root, text="Select Language:").pack(pady=10)
        self.lang_combo = ttk.Combobox(self.root, width=40)
        self.lang_combo['values'] = ["es", "en", "fr", "de", "it", "pt", "auto"]
        self.lang_combo.pack(pady=5)
        self.lang_combo.set(self.config.get("language", "es"))

        # Translation Language Selection
        ttk.Label(self.root, text="Translation Target Language (Click to Translate):").pack(pady=10)
        self.trans_lang_combo = ttk.Combobox(self.root, width=40)
        self.trans_lang_combo['values'] = ["en", "es", "fr", "de", "it", "pt"]
        self.trans_lang_combo.pack(pady=5)
        self.trans_lang_combo.set(self.config.get("translation_lang", "en"))

        # Start Button
        btn_text = "Restart Captions" if self.restart_callback else "Start Captions"
        ttk.Button(self.root, text=btn_text, command=self.start_app).pack(pady=30)
        
        self.root.mainloop()

    def start_app(self):
        selected_device_name = self.device_combo.get()
        selected_model = self.model_combo.get()
        selected_lang = self.lang_combo.get()
        selected_trans_lang = self.trans_lang_combo.get()
        
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
        save_config(selected_device_name, selected_model, selected_lang, selected_trans_lang)

        # Destroy config window
        try:
             self.root.destroy()
        except:
             pass
             
        if self.restart_callback:
            self.restart_callback(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang)
        else:
            CaptionWindow(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang)

