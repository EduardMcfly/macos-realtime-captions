import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import os
import sounddevice as sd
from screeninfo import get_monitors

from .app_config import load_config, save_config, stop_event, CONFIG_FILE, pause_event, transcription_paused
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
        
        # Track config window instance
        self.config_window = None
        
        # Translation State
        self.translation_lang = translation_lang
        self.translation_model = None
        self.translation_tokenizer = None
        self.is_translating = False
        
        # Window Dimensions (initial hint)
        window_width = 800
        window_height = 200
        padding_bottom = 50

        # Apply initial size FIRST
        self.root.geometry(f"{window_width}x{window_height}")
        # Ensure Tk layout is ready
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
            print("🧠 Pre-loading Translation Model (Llama-3.2-1B)...")
            from mlx_lm import load
            # Switching to Llama-3.2-1B-Instruct-4bit for better stability/compatibility
            self.translation_model, self.translation_tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
            print("🧠 Translation Model Ready!")
        except Exception as e:
            print(f"⚠️ Failed to load translation model: {e}")

    def on_text_click(self, event):
        """Handle click to translate current text."""
        # Find which tag/segment was clicked
        try:
            index = self.text_area.index(f"@{event.x},{event.y}")
            tags = self.text_area.tag_names(index)
            
            # Look for our segment tag
            target_seg_id = None
            for tag in tags:
                if tag.startswith("seg_"):
                    target_seg_id = tag
                    break
            
            if not target_seg_id:
                return # Clicked on whitespace or non-segment
            
            # Check if already translating this specific segment
            # We can use a set or check if a translation tag exists
            trans_tag = f"trans_{target_seg_id}"
            if self.text_area.tag_ranges(trans_tag):
                print("Already translated/translating this segment.")
                return

            threading.Thread(target=self.perform_translation, args=(target_seg_id,), daemon=True).start()
            
        except Exception as e:
            print(f"Click Error: {e}")

    def perform_translation(self, seg_id):
        # PAUSE TRANSCRIPTION to prevent Metal/MLX concurrency crash
        pause_event.set()
        
        # Wait for transcriber to acknowledge pause (max 3s)
        # This ensures Whisper has finished its current pass and released GPU resources
        if not transcription_paused.wait(timeout=3.0):
             print("⚠️ Warning: Transcriber did not pause in time. Proceeding with caution...")
        
        try:
            # Get text content of the segment
            start, end = self.text_area.tag_ranges(seg_id)
            segment_text = self.text_area.get(start, end).strip()
            
            if not segment_text: return

            # Insert a temporary "Translating..." placeholder immediately after the segment
            # We schedule this on the UI thread
            self.schedule_insert_placeholder(seg_id)

            print(f"🔄 Translating segment '{segment_text[:20]}...' to {self.translation_lang}...")
            # self.schedule_set_status(f"🔄 Translating to {self.translation_lang}...")

            if not self.translation_model:
                self.preload_translation_model()

            from mlx_lm import generate
            
            # Construct Prompt
            # Explicitly state the target language name for clarity
            lang_map = {
                "en": "English", "es": "Spanish", "fr": "French", 
                "de": "German", "it": "Italian", "pt": "Portuguese"
            }
            target_lang_name = lang_map.get(self.translation_lang, self.translation_lang)
            
            messages = [
                {"role": "system", "content": f"You are a professional translator. Translate the following text strictly into {target_lang_name}. Do not add explanations, just the translation."},
                {"role": "user", "content": segment_text}
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
            
            # Update UI with translation inline
            self.schedule_insert_translation(seg_id, translation.strip())

        except Exception as e:
            print(f"Translation Error: {e}")
            self.schedule_set_status(f"❌ Translation Error: {e}")
            # Remove placeholder if failed? (Enhancement)
        finally:
            # RESUME TRANSCRIPTION
            pause_event.clear()
        
    def schedule_insert_placeholder(self, seg_id):
        self.root.after(0, self.insert_placeholder, seg_id)

    def insert_placeholder(self, seg_id):
        try:
            self.text_area.config(state="normal")
            
            # Find end of segment
            # tag_ranges returns flat list (start, end, start, end...) usually just 2 for us
            ranges = self.text_area.tag_ranges(seg_id)
            if not ranges: return
            end_index = ranges[-1]
            
            trans_id = f"trans_{seg_id}"
            
            # Insert loading text
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
            
            # Find the placeholder range
            ranges = self.text_area.tag_ranges(trans_id)
            if ranges:
                # Delete placeholder
                self.text_area.delete(ranges[0], ranges[1])
                # Insert actual translation at the same spot (ranges[0])
                # We add a small indentation or visual separation
                final_text = f"\n↳ {text}" 
                self.text_area.insert(ranges[0], final_text, trans_id)
                # Update tag style
                self.text_area.tag_config(trans_id, font=("Helvetica", 18, "italic"), foreground="#aaaaaa")
            
            self.text_area.config(state="disabled")
        except: pass

    def open_settings(self):
        # Prevent multiple windows
        if self.config_window is not None and self.config_window.root.winfo_exists():
            self.config_window.root.lift()
            self.config_window.root.focus_force()
            return
            
        # Stop processing but keep window open
        stop_event.set()
        
        # Open config window as Toplevel of this root
        # We capture the instance
        self.config_window = ConfigWindow(parent=self.root, restart_callback=self.restart_processing, on_close_callback=self.on_config_close)
        
    def on_config_close(self):
        # Callback when config window closes without restarting
        self.config_window = None
        # Resume? Maybe not automatically if user cancelled.
        # But if they just closed it, we should probably ensure stop_event is cleared if we want to resume?
        # Current logic stops processing when opening settings. 
        # If user cancels settings, they probably expect it to resume or stay stopped?
        # Let's assume they want to resume if they didn't change anything.
        stop_event.clear()
        
        # We need to restart the thread if it died.
        if not self.processing_thread.is_alive():
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

    def restart_processing(self, model_size, device_index, device_name, language, translation_lang):
        print(f"🔄 Restarting with Translation Lang: {translation_lang}")
        
        # Clear config window ref
        self.config_window = None
        
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
                    # Create a unique tag for this segment to allow individual translation
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
            
            # Don't clear previous text!
            # self.text_area.delete("1.0", tk.END)
            
            # Append system message
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
            # Register this window with the parent CaptionWindow if possible
            # We assume parent is CaptionWindow.root. 
            # We can find the CaptionWindow instance via some hack or just expect CaptionWindow to track it via the on_close_callback or similar.
            # Actually, let's just use the fact that 'parent' is the root widget.
            # We can't easily set self.config_window on the CaptionWindow instance from here without passing the instance.
            # So we rely on CaptionWindow checking 'winfo_exists' if it kept a reference? 
            # Wait, CaptionWindow.open_settings didn't capture the instance because __init__ doesn't return it.
            # We need to fix that interaction.
            
            # Better approach:
            # We set the 'config_window' attribute of the CaptionWindow app if we can find it, 
            # OR we just let CaptionWindow handle the reference if we return this object. 
            # But we are inside __init__.
            
            # Let's set it as a transient window
            self.root.transient(parent)
        else:
            self.root = tk.Tk()
            
        self.root.title("Setup Live Captions")
        self.root.geometry("400x450")
        
        # Center window logic
        if parent:
             # Center relative to parent
             try:
                 parent_x = parent.winfo_rootx()
                 parent_y = parent.winfo_rooty()
                 parent_w = parent.winfo_width()
                 parent_h = parent.winfo_height()
                 
                 my_w = 400
                 my_h = 450
                 
                 x = parent_x + (parent_w - my_w) // 2
                 y = parent_y - my_h - 20 # Above the parent window
                 
                 # Keep on screen
                 if y < 0: y = parent_y + parent_h + 20
                 
                 self.root.geometry(f"+{x}+{y}")
             except: pass
        
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
        
        # Explicitly update the instance config so it persists in this session
        self.config["translation_lang"] = selected_trans_lang
        self.config["language"] = selected_lang
        self.config["model_size"] = selected_model
        self.config["device_name"] = selected_device_name

        # Destroy config window
        try:
             self.root.destroy()
        except:
             pass
             
        if self.restart_callback:
            self.restart_callback(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang)
        else:
            CaptionWindow(selected_model, device_index, selected_device_name, selected_lang, selected_trans_lang)

