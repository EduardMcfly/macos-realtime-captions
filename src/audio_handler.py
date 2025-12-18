import sounddevice as sd
import queue
import threading
from typing import List, Dict, Any, Optional
import numpy as np

class AudioRecorder:
    def __init__(self, input_device_index: int, sample_rate: int, audio_queue: queue.Queue):
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.audio_queue = audio_queue
        self.stream: Optional[sd.InputStream] = None
        self.running = False

    def start(self):
        """Starts the audio recording stream."""
        if self.running:
            return

        self.running = True
        self.stream = sd.InputStream(
            device=self.input_device_index,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stops the audio recording stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any):
        """Callback for sounddevice."""
        if self.running:
            self.audio_queue.put(indata.copy())
        if status:
            print(f"Audio status: {status}")

    @staticmethod
    def get_audio_devices() -> List[Dict[str, Any]]:
        """Returns a list of available audio devices."""
        try:
            return sd.query_devices()
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            return []
