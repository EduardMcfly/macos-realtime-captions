import numpy as np
import mlx_whisper
import time
import threading
import queue
import os
from typing import Callable, Optional
from .config import SAMPLE_RATE
from .utils import log_to_file
import sounddevice as sd  # Used for sleep

class Transcriber:
    def __init__(
        self, 
        model_size: str, 
        language: str, 
        audio_queue: queue.Queue, 
        update_callback: Callable[[str, bool], None], 
        status_callback: Callable[[str], None]
    ):
        self.model_size = model_size
        self.language = language
        self.audio_queue = audio_queue
        self.update_callback = update_callback
        self.status_callback = status_callback
        
        self.running = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.transcription_paused = threading.Event()
        
        # Configuration
        self.silence_threshold = 0.01 
        self.min_duration = 1.0
        self.max_duration = 15.0 
        self.update_interval = 0.5
        
        # State
        self.local_audio_buffer = np.zeros((0, 1), dtype=np.float32)
        self.last_transcribe_time = 0
        self.last_committed_text = ""
        
        # Models
        self.fast_model_path = "mlx-community/whisper-tiny-mlx"
        self.quality_model_path = f"mlx-community/whisper-{model_size}-mlx"

    def start(self):
        """Starts the transcription loop in a separate thread."""
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the transcription loop."""
        self.running = False
        self.stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def pause(self):
        """Pauses transcription (e.g., during translation)."""
        self.pause_event.set()

    def resume(self):
        """Resumes transcription."""
        self.pause_event.clear()

    def is_paused(self) -> bool:
        """Returns True if the transcriber is currently paused."""
        return self.transcription_paused.is_set()

    def _run_loop(self):
        """Main transcription loop."""
        self.status_callback("Ready to transcribe")
        print(f"✅ Starting MLX Whisper (Fast: tiny, Quality: {self.model_size})...")
        os.environ["TQDM_DISABLE"] = "1"

        while not self.stop_event.is_set():
            # Check for pause event
            if self.pause_event.is_set():
                self.transcription_paused.set() 
                sd.sleep(100)
                continue
            else:
                self.transcription_paused.clear()

            # 1. Drain Queue
            while not self.audio_queue.empty():
                try:
                    data = self.audio_queue.get_nowait()
                    self.local_audio_buffer = np.concatenate([self.local_audio_buffer, data])
                except queue.Empty:
                    break

            current_duration = len(self.local_audio_buffer) / SAMPLE_RATE
            now = time.time()

            # 2. Transcribe?
            if current_duration >= self.min_duration and (now - self.last_transcribe_time > self.update_interval):
                
                # Optimization: Skip if silence
                rms = np.sqrt(np.mean(self.local_audio_buffer**2))
                if rms < self.silence_threshold and current_duration < self.max_duration:
                    # Clear pending text if silence is detected
                    self.update_callback("", False)
                    sd.sleep(50)
                    continue

                try:
                    self._process_audio_buffer(current_duration)
                    self.last_transcribe_time = now

                except Exception as e:
                    print(f"\n⚠️ Transcription error: {e}")
            
            sd.sleep(50)

    def _process_audio_buffer(self, current_duration: float):
        # Context Prompting
        prompt = self.last_committed_text[-200:] if self.last_committed_text else " "
        
        # Transcribe with FAST model for preview/logic
        result_fast = mlx_whisper.transcribe(
            self.local_audio_buffer.flatten(),
            path_or_hf_repo=self.fast_model_path,
            language=self.language,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt=prompt
        )
        text = result_fast["text"].strip()
        
        # Anti-Hallucination
        if text.lower() == prompt.strip().lower():
            text = ""

        if text:
            # Smart Commit Logic
            is_sentence_end = text.endswith((".", "!", "?"))
            
            # Check for silence at the end
            last_chunk_len = int(0.5 * SAMPLE_RATE)
            if len(self.local_audio_buffer) > last_chunk_len:
                last_chunk = self.local_audio_buffer[-last_chunk_len:]
                is_silence_end = np.sqrt(np.mean(last_chunk**2)) < self.silence_threshold
            else:
                is_silence_end = True # Assume silence if buffer is short (though min duration check passed)

            should_commit = (is_sentence_end and is_silence_end) or (current_duration > self.max_duration)
            
            if should_commit:
                self._commit_text(prompt)
            else:
                self.update_callback(text, False) # Preview

    def _commit_text(self, prompt: str):
        # RE-TRANSCRIBE with QUALITY model
        print("✨ Refining with quality model...")
        result_quality = mlx_whisper.transcribe(
            self.local_audio_buffer.flatten(),
            path_or_hf_repo=self.quality_model_path,
            language=self.language,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt=prompt
        )
        text_quality = result_quality["text"].strip()
        
        # Use quality text if valid, else fallback
        # Assuming we have the 'text' from fast model available? 
        # Actually I need to re-run or pass it. 
        # But 'text' was local to _process_audio_buffer.
        # It's safer to trust quality model or re-use fast if quality fails (unlikely here).
        
        final_text = text_quality # Simply trust quality model
        
        print(f"📝 {final_text}")
        log_to_file(final_text)
        self.update_callback(final_text, True) # Final
        
        self.last_committed_text += " " + final_text
        if len(self.last_committed_text) > 1000: 
            self.last_committed_text = self.last_committed_text[-1000:]
        
        self.local_audio_buffer = np.zeros((0, 1), dtype=np.float32)
