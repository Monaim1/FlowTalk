"""
Whisper ASR Module

Speech-to-text transcription using OpenAI's Whisper model via HuggingFace.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class TranscriptSegment:
    """Container for a transcript segment with timing."""
    text: str
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    confidence: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_time - self.start_time


class WhisperASR:
    """
    Speech recognition using Whisper.
    
    Transcribes audio to text with word/segment timestamps.
    
    Usage:
        asr = WhisperASR(model_name="openai/whisper-small")
        segments = asr.transcribe(audio_chunk)
        for seg in segments:
            print(f"[{seg.start_time:.2f}s] {seg.text}")
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: Optional[str] = None,
        use_gpu: bool = True,
        return_timestamps: bool = True,
    ):
        """
        Initialize Whisper ASR.
        
        Args:
            model_name: HuggingFace model name (whisper-base, whisper-small, whisper-medium)
            language: Language code (None for auto-detect)
            use_gpu: Whether to use GPU acceleration
            return_timestamps: Whether to return word/segment timestamps
        """
        self.model_name = model_name
        self.language = language
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.return_timestamps = return_timestamps
        
        self._pipeline = None
        self._device = "cuda" if self.use_gpu else "cpu"
        
    def _load_pipeline(self):
        """Lazy-load the ASR pipeline."""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline
            
            print(f"[WhisperASR] Loading model: {self.model_name}")
            
            # Create pipeline with appropriate settings
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self._device,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
            )
            
            print(f"[WhisperASR] Model loaded on {self._device}")
            
        except Exception as e:
            print(f"[WhisperASR] Failed to load model: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_offset: float = 0.0,
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio to text with timestamps.
        
        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Audio sample rate (should be 16000 for Whisper)
            chunk_offset: Time offset to add to segment timestamps
        
        Returns:
            List of TranscriptSegment objects
        """
        self._load_pipeline()
        
        if len(audio) == 0:
            return []
        
        # Ensure proper format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        try:
            # Prepare generate kwargs
            generate_kwargs = {}
            if self.language:
                generate_kwargs["language"] = self.language
                generate_kwargs["task"] = "transcribe"
            
            # Run transcription
            result = self._pipeline(
                {"sampling_rate": sample_rate, "raw": audio},
                return_timestamps=self.return_timestamps,
                generate_kwargs=generate_kwargs,
            )
            
            segments = []
            
            if self.return_timestamps and "chunks" in result:
                # Process timestamped chunks
                for chunk in result["chunks"]:
                    text = chunk.get("text", "").strip()
                    if not text:
                        continue
                    
                    timestamps = chunk.get("timestamp", (0, 0))
                    start_time = (timestamps[0] or 0) + chunk_offset
                    end_time = (timestamps[1] or start_time) + chunk_offset
                    
                    segments.append(TranscriptSegment(
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    ))
            else:
                # Single segment for entire audio
                text = result.get("text", "").strip()
                if text:
                    duration = len(audio) / sample_rate
                    segments.append(TranscriptSegment(
                        text=text,
                        start_time=chunk_offset,
                        end_time=chunk_offset + duration,
                    ))
            
            return segments
            
        except Exception as e:
            print(f"[WhisperASR] Transcription error: {e}")
            return []
    
    def transcribe_simple(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Simple transcription returning just text (no timestamps).
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
        
        Returns:
            Transcribed text string
        """
        segments = self.transcribe(audio, sample_rate)
        return " ".join(seg.text for seg in segments)
    
    def transcribe_file(self, file_path: str) -> List[TranscriptSegment]:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            List of TranscriptSegment objects
        """
        self._load_pipeline()
        
        try:
            result = self._pipeline(
                file_path,
                return_timestamps=self.return_timestamps,
            )
            
            segments = []
            
            if self.return_timestamps and "chunks" in result:
                for chunk in result["chunks"]:
                    text = chunk.get("text", "").strip()
                    if not text:
                        continue
                    
                    timestamps = chunk.get("timestamp", (0, 0))
                    segments.append(TranscriptSegment(
                        text=text,
                        start_time=timestamps[0] or 0,
                        end_time=timestamps[1] or 0,
                    ))
            else:
                text = result.get("text", "").strip()
                if text:
                    segments.append(TranscriptSegment(
                        text=text,
                        start_time=0,
                        end_time=0,  # Unknown duration
                    ))
            
            return segments
            
        except Exception as e:
            print(f"[WhisperASR] File transcription error: {e}")
            return []


class MockASR:
    """
    Mock ASR for testing without model loading.
    
    Returns placeholder transcriptions.
    """
    
    def __init__(self, **kwargs):
        pass
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_offset: float = 0.0,
    ) -> List[TranscriptSegment]:
        """Return mock transcript."""
        duration = len(audio) / sample_rate
        return [
            TranscriptSegment(
                text="[Mock transcription]",
                start_time=chunk_offset,
                end_time=chunk_offset + duration,
            )
        ]
    
    def transcribe_simple(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        return "[Mock transcription]"
    
    def transcribe_file(self, file_path: str) -> List[TranscriptSegment]:
        return [
            TranscriptSegment(
                text="[Mock file transcription]",
                start_time=0,
                end_time=0,
            )
        ]
