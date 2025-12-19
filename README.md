# Live Captions for macOS (MLX Optimized)

A real-time captioning application optimized for Apple Silicon (M1/M2/M3) using `mlx-whisper`. It provides instant transcription previews using a lightweight model ("tiny") and commits finalized, high-quality text using a larger model (e.g., "medium" or "large") for accuracy.

## Features

- **Real-time Transcription**: Instant feedback using a fast preview model.
- **Dual-Model Architecture**:
  - **Fast Path**: Uses `whisper-tiny` for low-latency previews.
  - **Quality Path**: Uses `whisper-medium` (or configured size) for final commits.
- **Smart Commit Strategy**: Commits text naturally based on sentence endings and silence detection.
- **Context-Aware**: Uses previous context to reduce hallucinations and improve continuity.
- **macOS Optimized**: Built on Apple's MLX framework for high-performance inference on Apple Silicon.
- **Customizable**: Select microphone/input device, model size, and language via a GUI.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3).
- **Python 3.10+** (Recommended).
- `ffmpeg` installed (required by `faster-whisper`/`mlx-whisper` for audio processing).

### Install ffmpeg
```bash
brew install ffmpeg
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/EduardMcfly/macos-realtime-captions.git
    cd macos-realtime-captions
    ```

2.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    python main.py
    ```

2.  **Configuration**:
    - Upon launch, a settings window will appear.
    - **Audio Device**: Select your microphone or "BlackHole" (for capturing system audio).
    - **Model Size**: Select the model for the "Quality" path (e.g., `medium`, `large-v3`). The preview always uses `tiny`.
    - **Language**: Choose the spoken language or `auto`.

3.  **Captions**:
    - The main window will appear at the bottom of your screen.
    - Start speaking! Text will appear in gray (preview) and turn white when finalized.

## Troubleshooting

- **No Audio/Captions**: Ensure you selected the correct input device. If using BlackHole/Loopback, ensure audio is actually being routed to it.
- **Slow Performance**: MLX is fast, but larger models (`large-v3`) still require significant memory and compute. Try `small` or `medium` if latency is high.
- **Permissions**: Ensure your terminal or IDE has permission to access the Microphone in macOS System Settings.

## Project Structure

- `main.py`: Entry point.
- `app_config.py`: Shared constants and configuration.
- `audio_handler.py`: Microphone input handling.
- `transcriber.py`: Core transcription logic (Dual-Model streaming).
- `ui.py`: Tkinter-based GUI.
- `utils.py`: Logging and helpers.

