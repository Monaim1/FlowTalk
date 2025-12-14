# FlowTalk 🌐

Real-Time Spoken Language Translator with speaker diarization.

## Features

- **Live Audio Capture**: Continuous microphone recording with VAD-based chunking
- **Speaker Diarization**: PyAnnote-powered speaker identification
- **Speech Recognition**: Whisper-based transcription with timestamps
- **Real-Time Translation**: Grradium API integration
- **Modern Web UI**: Gradio interface with speaker color-coding

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=your_huggingface_token
GRRADIUM_API_KEY=your_grradium_api_key
```

### 3. Run the Application

**Web UI (default):**
```bash
python main.py
```

**Console mode:**
```bash
python main.py --console
```

**Test mode (no API keys needed):**
```bash
python main.py --mock
```

## Usage Options

```
python main.py [OPTIONS]

Options:
  --source, -s    Source language code (default: en)
  --target, -t    Target language code (default: es)
  --model         Whisper model (tiny/base/small/medium)
  --mock          Test mode without API keys
  --console       Console mode instead of web UI
  --port          Server port (default: 7860)
  --share         Create public URL
  --no-gpu        Disable GPU acceleration
```

## Examples

```bash
# English to French
python main.py --source en --target fr

# Test mode
python main.py --mock

# Console with larger model
python main.py --console --model openai/whisper-medium

# Share publicly
python main.py --share
```

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Russian, Hindi, Dutch, Polish, Turkish

## Architecture

```
Microphone → Audio Buffer → [Diarization + ASR] → Alignment → Translation → UI
                              ↓ parallel ↓
                         Speaker labels + Transcript
```

## Requirements

- Python 3.9+
- Microphone access
- HuggingFace token (for PyAnnote)
- Grradium API key (for translation)
- GPU recommended for real-time performance

## License

MIT License
