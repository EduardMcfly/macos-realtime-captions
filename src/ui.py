import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import queue
import time
from typing import Optional

from .config import ConfigManager
from .audio_handler import AudioRecorder
from .transcriber import Transcriber
from screeninfo import get_monitors

class CaptionWindow:
    def __init__(
        self, 
        model_size: str, 
        device_index: int, 
        device_name: str, 
        language: str, 
        translation_lang: str = "en", 
        translation_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    ):
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
        
        # Track config window instance
        self.config_window = None
        
        # Translation State
        self.translation_lang = translation_lang
        self.translation_model_name = translation_model
        self.translation_model = None
        self.translation_tokenizer = None
        self.is_translating = False
        
        # Track active translations to prevent double-clicks
        self.active_translations = set()
        
        # Core Components
        self.audio_queue = queue.Queue()
        self.audio_recorder = AudioRecorder(device_index, 16000, self.audio_queue)
        
        self.model_size = model_size
        self.device_index = device_index
        self.device_name = device_name
        self.language = language if language != "auto" else None
        
        self.transcriber = Transcriber(
            model_size=self.model_size,
            language=self.language,
            audio_queue=self.audio_queue,
            update_callback=self.schedule_update_text,
            status_callback=self.schedule_set_status
        )

        # Window Dimensions (initial hint)
        self._setup_window_geometry()

        self.last_text_time = datetime.datetime.now()
        self.paragraph_threshold = 2.0 

        # UI Components
        config_btn = tk.Button(self.root, text="⚙️", font=("Arial", 14), bg="black", fg="white", bd=0, command=self.open_settings)
        config_btn.place(relx=1.0, x=-10, y=10, anchor="ne")

        self.text_area = tk.Text(self.root, font=("Helvetica", 20), fg="white", bg="black", wrap="word", height=5, bd=0, highlightthickness=0)
        self.text_area.pack(fill="both", expand=True, padx=15, pady=15)
        self.text_area.insert("1.0", "⏳ Loading Model... Please wait.\n")
        self.text_area.config(state="disabled")
        
        # Click to Translate binding
        self.text_area.bind("<Button-1>", self.on_text_click)
        
        # Start translation model loader in background
        threading.Thread(target=self.preload_translation_model, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Schedule processing start after mainloop is running
        self.root.after(500, self.start_processing)
        
        # Start mainloop explicitly
        print("🖥️ Entering Main Loop...")
        self.root.mainloop()

    def start_processing(self):
        """Starts background threads once UI is ready."""
        self.transcriber.start()
        self.audio_recorder.start()

    def _setup_window_geometry(self):
        window_width = 800
        window_height = 200
        padding_bottom = 50

        # Apply initial size FIRST
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.update_idletasks()

        real_width = self.root.winfo_width()
        real_height = self.root.winfo_height()

        # Horizontal position → monitor with mouse
        mouse_x, mouse_y = self.root.winfo_pointerxy()
        monitors = get_monitors()
        target_monitor = None

        for m in monitors:
            if m.x <= mouse_x < m.x + m.width:
                target_monitor = m
                break

        if target_monitor is None and monitors:
            target_monitor = monitors[0]

        # X centered in monitor
        if target_monitor:
            x_pos = target_monitor.x + (target_monitor.width - real_width) // 2
        else:
            x_pos = (self.root.winfo_screenwidth() - real_width) // 2

        # Y using usable screen (dock-safe)
        y_pos = self.root.winfo_screenheight() - real_height - 20 - padding_bottom

        self.root.geometry(f"+{x_pos}+{y_pos}")

    def preload_translation_model(self):
        """Preload the lightweight LLM for translation to avoid lag on first click."""
        try:
            print(f"🧠 Pre-loading Translation Model ({self.translation_model_name})...")
            from mlx_lm import load
            self.translation_model, self.translation_tokenizer = load(self.translation_model_name)
            print("🧠 Translation Model Ready!")
        except Exception as e:
            print(f"⚠️ Failed to load translation model: {e}")

    def on_text_click(self, event):
        """Handle click to translate current text."""
        try:
            index = self.text_area.index(f"@{event.x},{event.y}")
            tags = self.text_area.tag_names(index)
            
            target_seg_id = None
            for tag in tags:
                if tag.startswith("seg_"):
                    target_seg_id = tag
                    break
            
            if not target_seg_id:
                return 
            
            # Allow re-translation even if it exists
            # We just need to remove the old one first if it exists to avoid duplication visual mess
            trans_tag = f"trans_{target_seg_id}"
            if self.text_area.tag_ranges(trans_tag):
                print("Re-translating segment...")
                # We don't return here anymore, we proceed to re-translate
            
            # Check if this segment is already being processed to avoid double-queueing
            if target_seg_id in self.active_translations:
                 print("Translation already in progress for this segment.")
                 return

            # --- Extract Text (Main Thread) ---
            # Search backwards for paragraph start (double newline)
            # We use the segment's END as reference to ensure we catch the start of THIS paragraph
            # (If we use start, and start is \n\n, backward search might skip it and find previous paragraph)
            seg_end_idx = self.text_area.index(f"{target_seg_id}.last")
            
            para_start_idx = self.text_area.search("\n\n", seg_end_idx, backwards=True, stopindex="1.0")
            if not para_start_idx:
                para_start_idx = "1.0"
            else:
                para_start_idx = f"{para_start_idx} + 2 chars"
            
            # Search forwards for paragraph end
            # We search from segment END to find the NEXT separator
            para_end_idx = self.text_area.search("\n\n", seg_end_idx, stopindex="end")
            if not para_end_idx:
                para_end_idx = "end-1c"
            
            # Get full paragraph text
            full_text = self.text_area.get(para_start_idx, para_end_idx).strip()
            
            # Filter
            clean_lines = [
                line for line in full_text.split('\n') 
                if not line.strip().startswith("↳") 
                and not line.strip().startswith("(Translating...") 
                and "[System" not in line
            ]
            segment_text = " ".join(clean_lines).strip()
            
            if not segment_text: 
                print("Empty segment text, skipping.")
                return

            # --- Lock & Feedback (Main Thread) ---
            self.active_translations.add(target_seg_id)
            
            # If re-translating, we might want to clear old translation visually first or just overwrite it
            # The insert_placeholder will handle insertion, but if "trans_id" exists it might just append?
            # insert_placeholder uses insert, but doesn't delete old unless we tell it to.
            # Actually insert_placeholder just inserts. 
            # If we already have a translation, we should probably delete it now to show "Translating..." again.
            if self.text_area.tag_ranges(trans_tag):
                ranges = self.text_area.tag_ranges(trans_tag)
                self.text_area.config(state="normal")
                self.text_area.delete(ranges[0], ranges[1])
                self.text_area.config(state="disabled")

            self.insert_placeholder(target_seg_id) # Immediate visual feedback

            # --- Start Background Task ---
            print(f"🚀 Starting translation for: {segment_text[:30]}...")
            threading.Thread(
                target=self.perform_translation, 
                args=(target_seg_id, segment_text), 
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"Click Error: {e}")
            if target_seg_id and target_seg_id in self.active_translations:
                 self.active_translations.remove(target_seg_id)

    def perform_translation(self, seg_id, text):
        # PAUSE TRANSCRIPTION
        self.transcriber.pause()
        
        start_wait = time.time()
        while not self.transcriber.is_paused() and (time.time() - start_wait < 3.0):
            time.sleep(0.1)
            
        try:
            print(f"🔄 Translating '{text[:20]}...' to {self.translation_lang}...")
            
            if not self.translation_model:
                self.schedule_set_status("⏳ Loading Translation Model...")
                self.preload_translation_model()

            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            
            lang_map = {
                "en": "English", "es": "Spanish", "fr": "French", 
                "de": "German", "it": "Italian", "pt": "Portuguese"
            }
            target_lang_name = lang_map.get(self.translation_lang, self.translation_lang)
            
            messages = [
                {"role": "system", "content": f"You are a professional interpreter. Translate the exact text provided below into {target_lang_name}. Translate every single sentence found in the input. Do not summarize, do not omit any details. Do not add explanations, just output the translation."},
                {"role": "user", "content": f"'''{text}'''"}
            ]
            
            prompt = self.translation_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            sampler = make_sampler(temp=0.1)

            translation = generate(
                self.translation_model, 
                self.translation_tokenizer, 
                prompt=prompt, 
                max_tokens=500, 
                verbose=False,
                sampler=sampler
            )
            
            self.schedule_insert_translation(seg_id, translation.strip())
            
        except Exception as e:
            print(f"Translation Error: {e}")
            self.schedule_set_status(f"❌ Translation Error: {e}")
            # Optionally remove placeholder or show error in text?
        finally:
            self.schedule_cleanup_lock(seg_id)
            self.transcriber.resume()

    def schedule_cleanup_lock(self, seg_id):
        self.root.after(0, self.cleanup_lock, seg_id)

    def cleanup_lock(self, seg_id):
        if seg_id in self.active_translations:
            self.active_translations.remove(seg_id)
        
    def schedule_insert_placeholder(self, seg_id):
        self.root.after(0, self.insert_placeholder, seg_id)

    def insert_placeholder(self, seg_id):
        try:
            self.text_area.config(state="normal")
            ranges = self.text_area.tag_ranges(seg_id)
            if not ranges: return
            end_index = ranges[-1]
            
            trans_id = f"trans_{seg_id}"
            self.text_area.insert(end_index, "\n(Translating...)", trans_id)
            self.text_area.tag_config(trans_id, font=("Helvetica", 14, "italic"), foreground="#888888")
            self.text_area.config(state="disabled")
        except: pass

    def schedule_insert_translation(self, seg_id, text):
        self.root.after(0, self.insert_translation, seg_id, text)

    def insert_translation(self, seg_id, text):
        try:
            self.text_area.config(state="normal")
            trans_id = f"trans_{seg_id}"
            ranges = self.text_area.tag_ranges(trans_id)
            if ranges:
                self.text_area.delete(ranges[0], ranges[1])
                final_text = f"\n↳ {text}" 
                self.text_area.insert(ranges[0], final_text, trans_id)
                self.text_area.tag_config(trans_id, font=("Helvetica", 18, "italic"), foreground="#aaaaaa")
            self.text_area.config(state="disabled")
        except: pass

    def open_settings(self):
        if self.config_window is not None and self.config_window.root.winfo_exists():
            self.config_window.root.lift()
            self.config_window.root.focus_force()
            return
            
        self.transcriber.stop()
        self.audio_recorder.stop()
        
        self.config_window = ConfigWindow(parent=self.root, restart_callback=self.restart_processing, on_close_callback=self.on_config_close)
        
    def on_config_close(self):
        self.config_window = None
        # Resume processing
        self.transcriber.start()
        self.audio_recorder.start()

    def restart_processing(self, model_size, device_index, device_name, language, translation_lang, translation_model):
        print(f"🔄 Restarting with Translation Lang: {translation_lang}, Model: {translation_model}")
        
        self.config_window = None
        
        # Stop existing
        self.transcriber.stop()
        self.audio_recorder.stop()
        
        # Update State
        self.model_size = model_size
        self.device_index = device_index
        self.device_name = device_name
        self.language = language if language != "auto" else None
        self.translation_lang = translation_lang
        self.translation_model_name = translation_model
        
        # Reload translation model if it changed
        self.translation_model = None
        threading.Thread(target=self.preload_translation_model, daemon=True).start()
        
        # Update Window Title
        self.root.title(f"Live Captions - {device_name} ({model_size}) [{language} -> {translation_lang}]")
        
        # Re-initialize Core
        # Reuse queue? Yes.
        self.audio_recorder = AudioRecorder(device_index, 16000, self.audio_queue)
        
        self.transcriber = Transcriber(
            model_size=self.model_size,
            language=self.language,
            audio_queue=self.audio_queue,
            update_callback=self.schedule_update_text,
            status_callback=self.schedule_set_status
        )
        
        self.transcriber.start()
        self.audio_recorder.start()
        
        self.set_status(f"🔄 Restarted with {model_size}...")

    def on_close(self):
        self.transcriber.stop()
        self.audio_recorder.stop()
        try:
            self.root.destroy()
        except:
            pass
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
            was_at_bottom = self.text_area.yview()[1] > 0.99

            try:
                self.text_area.delete("pending.first", "pending.last")
            except tk.TclError:
                pass 
            
            if text:
                prefix = ""
                prev_char = self.text_area.get("end-2c", "end-1c")
                current_time = datetime.datetime.now()
                time_diff = (current_time - self.last_text_time).total_seconds()
                
                if time_diff > self.paragraph_threshold:
                    prefix = "\n\n"
                elif prev_char and prev_char not in [" ", "\n"] and not text.startswith(" "):
                    prefix = " "

                full_text = prefix + text
                
                if is_final:
                    seg_id = f"seg_{int(current_time.timestamp() * 1000)}"
                    self.text_area.insert(tk.END, full_text, seg_id)
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
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            msg = f"\n\n[System {timestamp}: {text}]\n\n"
            
            self.text_area.insert(tk.END, msg, "system_msg")
            self.text_area.tag_config("system_msg", foreground="#666666", font=("Helvetica", 12))
            
            self.text_area.see(tk.END)
            self.text_area.config(state="disabled")
        except: pass


class ConfigWindow:
    def __init__(self, parent=None, restart_callback=None, on_close_callback=None):
        self.restart_callback = restart_callback
        self.on_close_callback = on_close_callback
        self.parent = parent
        
        if parent:
            self.root = tk.Toplevel(parent)
            self.root.transient(parent)
        else:
            self.root = tk.Tk()

        my_w = 400
        my_h = 490
            
        self.root.title("Setup Live Captions")
        self.root.geometry(f"{my_w}x{my_h}")
        
        if parent:
             try:
                 parent_x = parent.winfo_rootx()
                 parent_y = parent.winfo_rooty()
                 parent_w = parent.winfo_width()
                 parent_h = parent.winfo_height()
                 x = parent_x + (parent_w - my_w) // 2
                 y = parent_y - my_h - 20 
                 if y < 0: y = parent_y + parent_h + 20
                 self.root.geometry(f"+{x}+{y}")
             except: pass
        
        self.config = ConfigManager.load_config()

        # Audio Device Selection
        ttk.Label(self.root, text="Select Audio Device (Microphone/BlackHole):").pack(pady=10)
        self.device_combo = ttk.Combobox(self.root, width=40)
        self.device_combo.pack(pady=5)
        
        self.devices = AudioRecorder.get_audio_devices()
        device_names = [d['name'] for d in self.devices]
        self.device_combo['values'] = device_names
        
        saved_device = self.config.get("device_name")
        if saved_device and saved_device in device_names:
             self.device_combo.set(saved_device)
        else:
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

        # Translation Model Selection
        ttk.Label(self.root, text="Translation AI Model:").pack(pady=10)
        self.trans_model_combo = ttk.Combobox(self.root, width=50)
        self.trans_model_combo['values'] = [
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/gemma-2-2b-it-4bit",
            "mlx-community/gemma-2-9b-it-4bit",
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
            "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
        ]
        self.trans_model_combo.pack(pady=5)
        self.trans_model_combo.set(self.config.get("translation_model", "mlx-community/gemma-2-2b-it-4bit"))
        
        ttk.Label(self.root, text="(Larger models = Better translation but slower)", font=("Arial", 10), foreground="gray").pack()

        # Start Button
        btn_text = "Save" if self.restart_callback else "Start Captions"
        ttk.Button(self.root, text=btn_text, command=self.start_app).pack(pady=30)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        if not parent:
            self.root.mainloop()

    def on_close(self):
        if self.on_close_callback:
            self.on_close_callback()
        try:
            self.root.destroy()
        except: pass

    def start_app(self):
        selected_device_name = self.device_combo.get()
        selected_model = self.model_combo.get()
        selected_lang = self.lang_combo.get()
        selected_trans_lang = self.trans_lang_combo.get()
        selected_trans_model = self.trans_model_combo.get()
        
        if not selected_device_name:
            messagebox.showwarning("Warning", "Please select an audio device.")
            return

        device_index = None
        for i, device in enumerate(self.devices):
            if device['name'] == selected_device_name:
                device_index = i
                break
        
        if device_index is None:
            messagebox.showerror("Error", "Selected device not found.")
            return

        ConfigManager.save_config(
            selected_device_name, 
            selected_model, 
            selected_lang, 
            selected_trans_lang, 
            selected_trans_model
        )
        
        try:
             self.root.destroy()
        except:
             pass
             
        if self.restart_callback:
            self.restart_callback(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang, selected_trans_model)
        else:
            CaptionWindow(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang, selected_trans_model)
