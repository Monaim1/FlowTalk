# FlowTalk 🌐

Real-time speech-to-speech translation powered by Gradium API.

## Features

- **Live Audio Capture**: Microphone streaming with voice activity detection
- **Speech-to-Text**: Real-time transcription via Gradium STT
- **Neural Translation**: Helsinki-NLP MarianMT models (EN↔FR)
- **Text-to-Speech**: Natural voice synthesis via Gradium TTS
- **Session Recording**: Automatic logging of transcripts to `records/`
- **Configurable Voice**: Custom TTS voice via environment variable

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```bash
GRADIUM_API_KEY=your_gradium_api_key
TTS_VOICE_ID=your_voice_id  # Optional, uses default if not set
```

### 3. Run the Translation Pipeline

```bash
uv run src/services/translation_pipeline.py
```

### 4. Run with Web UI (Optional)

```bash
uv run src/services/translation_ui.py
# Open http://127.0.0.1:7860
```

## Project Structure

```
FlowTalk/
├── src/services/
│   ├── audio_capture.py         # Microphone streaming
│   ├── translation_pipeline.py  # Main pipeline (CLI)
│   └── translation_ui.py        # Gradio web interface
├── records/                     # Session transcripts
├── .env                         # API keys (not committed)
└── pyproject.toml              # Dependencies
```

## Architecture

```
🎤 Microphone → STT (Gradium) → Translation (MarianMT) → TTS (Gradium) → 🔊 Speaker
                    ↓                    ↓
              Transcription         Translation
                    ↓                    ↓
                        → Recording (records/*.txt)
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GRADIUM_API_KEY` | Gradium API key (required) | - |
| `TTS_VOICE_ID` | Voice ID for TTS output | `wzsx4FjdulIY7oBs` |

## Supported Translation Directions

- **English → French**: `Helsinki-NLP/opus-mt-en-fr`
- **French → English**: `Helsinki-NLP/opus-mt-fr-en`

## Requirements

- Python 3.12+
- Microphone access
- Gradium API key
- macOS/Linux (Windows untested)

## License

MIT License
