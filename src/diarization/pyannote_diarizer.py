"""
PyAnnote Speaker Diarization Module

Identifies who is speaking when using PyAnnote's neural speaker diarization.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import torch
import io
import scipy.io.wavfile as wavfile


@dataclass
class SpeakerSegment:
    """Container for a speaker segment."""
    speaker_id: str       # Speaker identifier (e.g., "SPEAKER_0")
    speaker_index: int    # Numeric speaker index
    start_time: float     # Start time in seconds
    end_time: float       # End time in seconds
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_time - self.start_time


class PyAnnoteDiarizer:
    """
    Speaker diarization using PyAnnote.
    
    Identifies different speakers in audio and labels segments with speaker IDs.
    
    Usage:
        diarizer = PyAnnoteDiarizer()
        segments = diarizer.process(audio_chunk)
        for seg in segments:
            print(f"Speaker {seg.speaker_id}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
    """
    
    def __init__(
        self,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        use_gpu: bool = True,
        min_speakers: int = 1,
        max_speakers: int = 4,
    ):
        """
        Initialize PyAnnote diarizer.
        
        Args:
            pipeline_name: HuggingFace pipeline name
            hf_token: HuggingFace API token (required for gated models)
            use_gpu: Whether to use GPU acceleration
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers
        """
        self.pipeline_name = pipeline_name
        self.hf_token = hf_token
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        self._pipeline = None
        self._speaker_mapping: Dict[str, int] = {}  # Map pipeline IDs to consistent indices
        self._next_speaker_index = 0
        
    def _load_pipeline(self):
        """Lazy-load the diarization pipeline."""
        if self._pipeline is not None:
            return
        
        try:
            from pyannote.audio import Pipeline
            
            print(f"[PyAnnoteDiarizer] Loading pipeline: {self.pipeline_name}")
            self._pipeline = Pipeline.from_pretrained(
                self.pipeline_name,
                use_auth_token=self.hf_token
            )
            
            if self.use_gpu:
                self._pipeline.to(torch.device("cuda"))
                print("[PyAnnoteDiarizer] Using GPU acceleration")
            else:
                print("[PyAnnoteDiarizer] Using CPU")
                
        except Exception as e:
            print(f"[PyAnnoteDiarizer] Failed to load pipeline: {e}")
            raise
    
    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> io.BytesIO:
        """Convert numpy audio to WAV bytes for PyAnnote."""
        # Ensure proper format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize to int16 range for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to bytes buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        
        return buffer
    
    def _get_consistent_speaker_index(self, pipeline_speaker_id: str) -> int:
        """Map pipeline speaker IDs to consistent indices across chunks."""
        if pipeline_speaker_id not in self._speaker_mapping:
            self._speaker_mapping[pipeline_speaker_id] = self._next_speaker_index
            self._next_speaker_index += 1
        return self._speaker_mapping[pipeline_speaker_id]
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_offset: float = 0.0,
    ) -> List[SpeakerSegment]:
        """
        Process audio chunk and return speaker segments.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate
            chunk_offset: Time offset to add to segment timestamps
        
        Returns:
            List of SpeakerSegment objects
        """
        self._load_pipeline()
        
        if len(audio) == 0:
            return []
        
        # Convert audio to format expected by PyAnnote
        wav_buffer = self._audio_to_wav_bytes(audio, sample_rate)
        
        try:
            # Run diarization with speaker count hints
            diarization = self._pipeline(
                wav_buffer,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            
            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_index = self._get_consistent_speaker_index(speaker)
                
                segment = SpeakerSegment(
                    speaker_id=f"Speaker_{speaker_index + 1}",  # 1-indexed for display
                    speaker_index=speaker_index,
                    start_time=turn.start + chunk_offset,
                    end_time=turn.end + chunk_offset,
                )
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            print(f"[PyAnnoteDiarizer] Processing error: {e}")
            return []
    
    def process_with_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_offset: float = 0.0,
    ) -> List[SpeakerSegment]:
        """
        Process audio with fallback to single speaker if diarization fails.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate
            chunk_offset: Time offset for timestamps
        
        Returns:
            List of SpeakerSegment objects (at least one default segment if failed)
        """
        try:
            segments = self.process(audio, sample_rate, chunk_offset)
            if segments:
                return segments
        except Exception as e:
            print(f"[PyAnnoteDiarizer] Fallback triggered: {e}")
        
        # Fallback: assume single speaker for entire chunk
        duration = len(audio) / sample_rate
        return [
            SpeakerSegment(
                speaker_id="Speaker_1",
                speaker_index=0,
                start_time=chunk_offset,
                end_time=chunk_offset + duration,
            )
        ]
    
    def reset_speaker_mapping(self):
        """Reset speaker ID mapping (e.g., for new session)."""
        self._speaker_mapping = {}
        self._next_speaker_index = 0
    
    def get_dominant_speaker(
        self,
        segments: List[SpeakerSegment],
        start_time: float,
        end_time: float,
    ) -> Optional[str]:
        """
        Get the dominant speaker for a time range based on overlap.
        
        Args:
            segments: List of speaker segments
            start_time: Start of range
            end_time: End of range
        
        Returns:
            Speaker ID with most overlap, or None if no overlap
        """
        speaker_durations: Dict[str, float] = {}
        
        for seg in segments:
            # Calculate overlap with range
            overlap_start = max(seg.start_time, start_time)
            overlap_end = min(seg.end_time, end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                speaker_durations[seg.speaker_id] = (
                    speaker_durations.get(seg.speaker_id, 0) + overlap_duration
                )
        
        if not speaker_durations:
            return None
        
        # Return speaker with most overlap
        return max(speaker_durations, key=speaker_durations.get)


class MockDiarizer:
    """
    Mock diarizer for testing without PyAnnote installation.
    
    Simply assigns all audio to a single speaker.
    """
    
    def __init__(self, **kwargs):
        self._speaker_index = 0
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_offset: float = 0.0,
    ) -> List[SpeakerSegment]:
        """Return single speaker segment for entire chunk."""
        duration = len(audio) / sample_rate
        return [
            SpeakerSegment(
                speaker_id="Speaker_1",
                speaker_index=0,
                start_time=chunk_offset,
                end_time=chunk_offset + duration,
            )
        ]
    
    def process_with_fallback(self, *args, **kwargs) -> List[SpeakerSegment]:
        return self.process(*args, **kwargs)
    
    def reset_speaker_mapping(self):
        pass
    
    def get_dominant_speaker(
        self,
        segments: List[SpeakerSegment],
        start_time: float,
        end_time: float,
    ) -> Optional[str]:
        if segments:
            return segments[0].speaker_id
        return None
