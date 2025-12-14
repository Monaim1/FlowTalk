"""
Audio Capture Module

Real-time microphone capture using sounddevice with thread-safe buffering.
"""
import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class AudioFrame:
    """Container for an audio frame with metadata."""
    data: np.ndarray
    timestamp: float  # Time since capture started
    sample_rate: int


class AudioCapture:
    """
    Real-time microphone capture with thread-safe frame buffering.
    
    Usage:
        capture = AudioCapture(sample_rate=16000)
        capture.start()
        # ... do processing ...
        frames = capture.get_frames()
        capture.stop()
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration: float = 0.02,  # 20ms frames
        device: Optional[int] = None,
        on_frame: Optional[Callable[[AudioFrame], None]] = None,
    ):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate in Hz (16000 for ASR compatibility)
            channels: Number of audio channels (1 for mono)
            frame_duration: Duration of each frame in seconds
            device: Audio device index (None for default)
            on_frame: Optional callback for each captured frame
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.device = device
        self.on_frame = on_frame
        
        # State
        self._stream: Optional[sd.InputStream] = None
        self._frame_queue: queue.Queue[AudioFrame] = queue.Queue()
        self._is_running = False
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags
    ):
        """Callback for sounddevice stream."""
        if status:
            # Log any stream issues but continue
            print(f"[AudioCapture] Stream status: {status}")
        
        if self._start_time is None:
            self._start_time = time.time()
        
        # Calculate timestamp relative to start
        timestamp = time.time() - self._start_time
        
        # Create audio frame
        frame = AudioFrame(
            data=indata.copy().flatten(),  # Copy to avoid buffer reuse issues
            timestamp=timestamp,
            sample_rate=self.sample_rate
        )
        
        # Add to queue
        self._frame_queue.put(frame)
        
        # Call optional callback
        if self.on_frame:
            try:
                self.on_frame(frame)
            except Exception as e:
                print(f"[AudioCapture] Callback error: {e}")
    
    def start(self):
        """Start audio capture."""
        with self._lock:
            if self._is_running:
                return
            
            self._is_running = True
            self._start_time = None
            
            # Clear any old frames
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Create and start stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.frame_size,
                device=self.device,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self._stream.start()
            print(f"[AudioCapture] Started (sample_rate={self.sample_rate}, frame_size={self.frame_size})")
    
    def stop(self):
        """Stop audio capture."""
        with self._lock:
            if not self._is_running:
                return
            
            self._is_running = False
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            print("[AudioCapture] Stopped")
    
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._is_running
    
    def get_frames(self, max_frames: Optional[int] = None) -> list[AudioFrame]:
        """
        Get captured frames from queue.
        
        Args:
            max_frames: Maximum number of frames to return (None for all)
        
        Returns:
            List of AudioFrame objects
        """
        frames = []
        count = 0
        
        while True:
            if max_frames and count >= max_frames:
                break
            try:
                frame = self._frame_queue.get_nowait()
                frames.append(frame)
                count += 1
            except queue.Empty:
                break
        
        return frames
    
    def get_frame_blocking(self, timeout: Optional[float] = None) -> Optional[AudioFrame]:
        """
        Get a single frame, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            AudioFrame or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    @property
    def queue_size(self) -> int:
        """Get current number of frames in queue."""
        return self._frame_queue.qsize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def list_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = []
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            devices.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            })
    return devices
