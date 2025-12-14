"""
Audio Buffer Module

Sliding window buffer with VAD-based chunking for real-time processing.
"""
import numpy as np
from typing import Optional, List, Callable
from dataclasses import dataclass
import time


@dataclass
class AudioChunk:
    """Container for a processed audio chunk with metadata."""
    data: np.ndarray
    start_time: float  # Start time relative to session
    end_time: float    # End time relative to session
    sample_rate: int
    is_speech: bool    # Whether speech was detected (from VAD)
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    def to_bytes(self) -> bytes:
        """Convert to bytes for processing."""
        return self.data.tobytes()


class AudioBuffer:
    """
    Sliding window audio buffer with Voice Activity Detection (VAD).
    
    Accumulates audio frames into chunks suitable for ASR/diarization processing.
    Uses energy-based VAD to detect speech and silence.
    
    Usage:
        buffer = AudioBuffer(chunk_duration=5.0, overlap=0.5)
        for frame in audio_frames:
            chunks = buffer.add_frame(frame.data, frame.timestamp)
            for chunk in chunks:
                process(chunk)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,      # Target chunk duration in seconds
        overlap_duration: float = 0.5,     # Overlap between chunks
        silence_threshold: float = 0.01,   # Energy threshold for speech detection
        silence_duration: float = 0.3,     # Seconds of silence to trigger chunk end
        min_chunk_duration: float = 0.5,   # Minimum chunk duration
        on_chunk: Optional[Callable[[AudioChunk], None]] = None,
    ):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Target duration for each chunk in seconds
            overlap_duration: Overlap between consecutive chunks
            silence_threshold: RMS energy threshold for speech detection
            silence_duration: Duration of silence to consider end of utterance
            min_chunk_duration: Minimum chunk duration before emitting
            on_chunk: Optional callback when chunk is ready
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_chunk_duration = min_chunk_duration
        self.on_chunk = on_chunk
        
        # Buffer state
        self._buffer: List[np.ndarray] = []
        self._buffer_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._total_samples = 0
        self._overlap_samples = int(overlap_duration * sample_rate)
        
    def _calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio signal."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def _is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech based on energy threshold."""
        energy = self._calculate_energy(audio)
        return energy > self.silence_threshold
    
    def _get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self._total_samples / self.sample_rate
    
    def _emit_chunk(self, end_time: float, force: bool = False) -> Optional[AudioChunk]:
        """
        Create and emit a chunk from buffer.
        
        Args:
            end_time: End timestamp for the chunk
            force: If True, emit even if below min duration
        
        Returns:
            AudioChunk if emitted, None otherwise
        """
        if not self._buffer:
            return None
        
        duration = self._get_buffer_duration()
        if not force and duration < self.min_chunk_duration:
            return None
        
        # Concatenate buffer
        audio_data = np.concatenate(self._buffer)
        
        # Check if chunk contains speech
        is_speech = self._is_speech(audio_data)
        
        # Create chunk
        chunk = AudioChunk(
            data=audio_data,
            start_time=self._buffer_start_time or 0.0,
            end_time=end_time,
            sample_rate=self.sample_rate,
            is_speech=is_speech
        )
        
        # Keep overlap for next chunk
        if self._overlap_samples > 0 and len(audio_data) > self._overlap_samples:
            overlap_data = audio_data[-self._overlap_samples:]
            self._buffer = [overlap_data]
            self._buffer_start_time = end_time - self.overlap_duration
            self._total_samples = len(overlap_data)
        else:
            self._buffer = []
            self._buffer_start_time = None
            self._total_samples = 0
        
        # Call callback if provided
        if self.on_chunk:
            try:
                self.on_chunk(chunk)
            except Exception as e:
                print(f"[AudioBuffer] Chunk callback error: {e}")
        
        return chunk
    
    def add_frame(
        self, 
        frame_data: np.ndarray, 
        timestamp: float
    ) -> List[AudioChunk]:
        """
        Add an audio frame to the buffer.
        
        Args:
            frame_data: Audio samples as numpy array
            timestamp: Frame timestamp (end of frame)
        
        Returns:
            List of ready chunks (usually 0 or 1)
        """
        chunks = []
        
        # Initialize buffer start time
        if self._buffer_start_time is None:
            frame_duration = len(frame_data) / self.sample_rate
            self._buffer_start_time = timestamp - frame_duration
        
        # Add frame to buffer
        self._buffer.append(frame_data)
        self._total_samples += len(frame_data)
        
        # Check for speech activity
        if self._is_speech(frame_data):
            self._last_speech_time = timestamp
        
        # Check if we should emit a chunk
        current_duration = self._get_buffer_duration()
        
        # Condition 1: Reached target duration
        if current_duration >= self.chunk_duration:
            chunk = self._emit_chunk(timestamp)
            if chunk:
                chunks.append(chunk)
        
        # Condition 2: Silence detected after speech (utterance end)
        elif self._last_speech_time is not None:
            silence_duration = timestamp - self._last_speech_time
            if silence_duration >= self.silence_duration and current_duration >= self.min_chunk_duration:
                chunk = self._emit_chunk(timestamp)
                if chunk:
                    chunks.append(chunk)
                self._last_speech_time = None  # Reset for next utterance
        
        return chunks
    
    def flush(self, end_time: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Flush remaining buffer as a final chunk.
        
        Args:
            end_time: End timestamp (uses current time if not provided)
        
        Returns:
            AudioChunk if buffer had content, None otherwise
        """
        if end_time is None:
            end_time = time.time()
        return self._emit_chunk(end_time, force=True)
    
    def reset(self):
        """Clear buffer state."""
        self._buffer = []
        self._buffer_start_time = None
        self._last_speech_time = None
        self._total_samples = 0
    
    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        return self._get_buffer_duration()
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0
