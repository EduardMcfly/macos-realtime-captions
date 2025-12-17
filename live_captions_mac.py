import queue
import sounddevice as sd
import numpy as np
import tkinter as tk
from faster_whisper import WhisperModel, download_model

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
BLOCK_SECONDS = 2   # lower = less delay
MODEL_SIZE = "medium"  # use "small" if Intel
# ----------------------------------------

audio_queue = queue.Queue()
audio_buffer = np.zeros((0, 1), dtype=np.float32)

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# -------- Floating Window --------
root = tk.Tk()
root.title("Live Captions")
root.attributes("-topmost", True)
root.attributes("-alpha", 0.88)
root.configure(bg="black")
root.overrideredirect(True)  # borderless

label = tk.Label(
    root,
    text="🎤 Listening...",
    font=("Helvetica", 26),
    fg="white",
    bg="black",
    wraplength=1000,
    justify="center"
)
label.pack(padx=25, pady=15)

# Initial position (bottom-center)
root.geometry("+200+650")

# Allow moving window with mouse
def start_move(event):
    root.x = event.x
    root.y = event.y

def stop_move(event):
    root.x = None
    root.y = None

def do_move(event):
    deltax = event.x - root.x
    deltay = event.y - root.y
    x = root.winfo_x() + deltax
    y = root.winfo_y() + deltay
    root.geometry(f"+{x}+{y}")

label.bind("<ButtonPress-1>", start_move)
label.bind("<ButtonRelease-1>", stop_move)
label.bind("<B1-Motion>", do_move)

# -------- Whisper --------
print(f"⏳ Verifying/Downloading model '{MODEL_SIZE}'...", flush=True)
model_path = download_model(MODEL_SIZE)  # Shows progress bar if downloading

print(f"✅ Model ready at: {model_path}", flush=True)
model = WhisperModel(
    model_path,
    device="cpu",
    compute_type="int8"
)

def process_audio():
    global audio_buffer

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
            label.config(text=text)

    root.after(200, process_audio)

# -------- Audio --------
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback
):
    print("🎤 Live captions active (Ctrl+C to exit)")
    process_audio()
    root.mainloop()
