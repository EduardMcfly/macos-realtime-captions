import numpy as np
import sounddevice as sd
import mlx_whisper
import time
import os
from app_config import stop_event, audio_queue, SAMPLE_RATE
from utils import log_to_file
from audio_handler import audio_callback

def run_transcription_loop(device_index, model_size, language, update_callback, status_callback):
    """
    Simplified single-loop transcriber.
    Accumulates audio and continuously transcribes the active buffer.
    Commits text when a sentence finishes or silence is detected.
    """
    
    # Internal State
    local_audio_buffer = np.zeros((0, 1), dtype=np.float32)
    last_transcribe_time = 0
    last_committed_text = ""
    
    # Configuration
    SILENCE_THRESHOLD = 0.01 
    MIN_DURATION = 1.0
    MAX_DURATION = 15.0 
    UPDATE_INTERVAL = 0.5

    try:
        status_callback("...")
        print(f"✅ Starting MLX Whisper ({model_size})...")
        os.environ["TQDM_DISABLE"] = "1"
        
        with sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback
        ):
            while not stop_event.is_set():
                # 1. Drain Queue
                while not audio_queue.empty():
                    local_audio_buffer = np.concatenate([local_audio_buffer, audio_queue.get()])

                current_duration = len(local_audio_buffer) / SAMPLE_RATE
                now = time.time()

                # 2. Transcribe?
                if current_duration >= MIN_DURATION and (now - last_transcribe_time > UPDATE_INTERVAL):
                    
                    # Optimization: Skip if silence
                    rms = np.sqrt(np.mean(local_audio_buffer**2))
                    if rms < SILENCE_THRESHOLD and current_duration < MAX_DURATION:
                        # Clear pending text if silence is detected to prevent stale previews
                        update_callback("", False)
                        sd.sleep(50)
                        continue

                    try:
                        # Context Prompting
                        prompt = last_committed_text[-200:] if last_committed_text else " "
                        
                        # Transcribe Full Buffer
                        result = mlx_whisper.transcribe(
                            local_audio_buffer.flatten(),
                            path_or_hf_repo=f"mlx-community/whisper-{model_size}-mlx",
                            language=language,
                            verbose=False,
                            temperature=0.0,
                            condition_on_previous_text=True,
                            initial_prompt=prompt
                        )
                        text = result["text"].strip()
                        
                        # Anti-Hallucination
                        if text.lower() == prompt.strip().lower():
                            text = ""

                        if text:
                            # 3. Smart Commit Logic
                            is_sentence_end = text.endswith((".", "!", "?"))
                            
                            # Check for silence at the end
                            last_chunk = local_audio_buffer[-int(0.5 * SAMPLE_RATE):] if len(local_audio_buffer) > 0.5 * SAMPLE_RATE else local_audio_buffer
                            is_silence_end = np.sqrt(np.mean(last_chunk**2)) < SILENCE_THRESHOLD

                            should_commit = (is_sentence_end and is_silence_end) or (current_duration > MAX_DURATION)
                            
                            if should_commit:
                                print(f"📝 {text}")
                                log_to_file(text)
                                update_callback(text, True) # Final
                                
                                last_committed_text += " " + text
                                if len(last_committed_text) > 1000: last_committed_text = last_committed_text[-1000:]
                                
                                local_audio_buffer = np.zeros((0, 1), dtype=np.float32)
                            else:
                                update_callback(text, False) # Preview

                        last_transcribe_time = now

                    except Exception as e:
                        print(f"\n⚠️ Transcription error: {e}")
                
                sd.sleep(50)

    except Exception as e:
        print(f"Critical Error: {e}")

