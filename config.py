"""
FlowTalk Configuration Module

Centralized configuration for API keys, model settings, and language options.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000  # 16kHz for ASR compatibility
    channels: int = 1  # Mono
    chunk_duration: float = 5.0  # Seconds per chunk
    overlap_duration: float = 0.5  # Overlap between chunks
    silence_threshold: float = 0.01  # Energy threshold for VAD
    silence_duration: float = 0.3  # Seconds of silence to end utterance
    frame_duration: float = 0.02  # 20ms frames


@dataclass
class DiarizationConfig:
    """PyAnnote diarization settings."""
    pipeline_name: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    min_speakers: int = 1
    max_speakers: int = 4
    use_gpu: bool = True


@dataclass
class ASRConfig:
    """Whisper ASR settings."""
    model_name: str = "openai/whisper-small"  # Options: whisper-base, whisper-small, whisper-medium
    language: Optional[str] = None  # None for auto-detect
    use_gpu: bool = True
    return_timestamps: bool = True


@dataclass
class TranslationConfig:
    """Grradium translation API settings."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GRRADIUM_API_KEY"))
    api_endpoint: str = os.getenv("GRRADIUM_API_ENDPOINT", "https://api.grradium.com/v1/translate")
    timeout: float = 10.0  # Request timeout in seconds
    max_retries: int = 1
    cache_enabled: bool = True


@dataclass
class UIConfig:
    """Gradio UI settings."""
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False  # Set True for public URL


# Supported languages for translation
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
}

# Speaker colors for UI display
SPEAKER_COLORS: List[str] = [
    "#3498db",  # Blue
    "#2ecc71",  # Green
    "#e74c3c",  # Red
    "#9b59b6",  # Purple
    "#f39c12",  # Orange
    "#1abc9c",  # Teal
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
]


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Default language settings
    source_language: str = "en"
    target_language: str = "es"


# Global config instance
config = Config()


def get_language_name(code: str) -> str:
    """Get language name from code."""
    return SUPPORTED_LANGUAGES.get(code, code)


def get_speaker_color(speaker_id: int) -> str:
    """Get color for a speaker ID (cycles through available colors)."""
    return SPEAKER_COLORS[speaker_id % len(SPEAKER_COLORS)]
