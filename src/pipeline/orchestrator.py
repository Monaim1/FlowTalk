"""
Pipeline Orchestrator

Coordinates all modules for real-time speech translation.
"""
import asyncio
import threading
import queue
from typing import Optional, Callable, List
from dataclasses import dataclass, field
import time
import numpy as np

from ..audio.capture import AudioCapture, AudioFrame
from ..audio.buffer import AudioBuffer, AudioChunk
from ..diarization.pyannote_diarizer import PyAnnoteDiarizer, SpeakerSegment, MockDiarizer
from ..transcription.whisper_asr import WhisperASR, TranscriptSegment, MockASR
from ..translation.grradium_translator import GrradiumTranslator, MockTranslator
from ..alignment.speaker_alignment import SpeakerAlignment, AlignedSegment


@dataclass
class TranslationEvent:
    """Event emitted when translation is ready."""
    speaker_id: str
    speaker_index: int
    original_text: str
    translated_text: str
    start_time: float
    end_time: float
    timestamp: float = field(default_factory=time.time)
    
    def format_timestamp(self) -> str:
        """Format start time as MM:SS."""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d}"


class PipelineOrchestrator:
    """
    Main coordinator for the speech translation pipeline.
    
    Manages concurrent processing of:
    - Audio capture
    - Speaker diarization
    - Speech recognition
    - Translation
    - UI updates
    
    Usage:
        pipeline = PipelineOrchestrator(...)
        pipeline.on_translation = callback_function
        pipeline.start()
        # ... later ...
        pipeline.stop()
    """
    
    def __init__(
        self,
        source_language: str = "en",
        target_language: str = "es",
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        overlap_duration: float = 0.5,
        use_gpu: bool = True,
        use_mock: bool = False,  # Use mock modules for testing
        hf_token: Optional[str] = None,
        grradium_api_key: Optional[str] = None,
        grradium_api_endpoint: Optional[str] = None,
        whisper_model: str = "openai/whisper-small",
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            sample_rate: Audio sample rate
            chunk_duration: Audio chunk duration in seconds
            overlap_duration: Overlap between chunks
            use_gpu: Use GPU acceleration if available
            use_mock: Use mock modules for testing
            hf_token: HuggingFace token for PyAnnote
            grradium_api_key: Grradium API key
            grradium_api_endpoint: Grradium API endpoint
            whisper_model: Whisper model name
        """
        self.source_language = source_language
        self.target_language = target_language
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.use_mock = use_mock
        
        # Initialize modules
        self._audio_capture = AudioCapture(
            sample_rate=sample_rate,
            frame_duration=0.02,
        )
        
        self._audio_buffer = AudioBuffer(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
        )
        
        # Use mock or real modules
        if use_mock:
            self._diarizer = MockDiarizer()
            self._asr = MockASR()
            self._translator = MockTranslator()
        else:
            self._diarizer = PyAnnoteDiarizer(
                hf_token=hf_token,
                use_gpu=use_gpu,
            )
            self._asr = WhisperASR(
                model_name=whisper_model,
                language=source_language,
                use_gpu=use_gpu,
            )
            self._translator = GrradiumTranslator(
                api_key=grradium_api_key,
                api_endpoint=grradium_api_endpoint or "https://api.grradium.com/v1/translate",
            )
        
        self._alignment = SpeakerAlignment()
        
        # Processing state
        self._is_running = False
        self._chunk_queue: queue.Queue[AudioChunk] = queue.Queue()
        self._result_queue: queue.Queue[TranslationEvent] = queue.Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Callbacks
        self.on_translation: Optional[Callable[[TranslationEvent], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        
        # Statistics
        self._chunks_processed = 0
        self._start_time: Optional[float] = None
    
    def _emit_status(self, status: str):
        """Emit status update."""
        if self.on_status:
            try:
                self.on_status(status)
            except Exception as e:
                print(f"[Pipeline] Status callback error: {e}")
    
    def _emit_error(self, error: str):
        """Emit error message."""
        print(f"[Pipeline] Error: {error}")
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                print(f"[Pipeline] Error callback error: {e}")
    
    def _emit_translation(self, event: TranslationEvent):
        """Emit translation event."""
        if self.on_translation:
            try:
                self.on_translation(event)
            except Exception as e:
                print(f"[Pipeline] Translation callback error: {e}")
    
    def _on_audio_chunk(self, chunk: AudioChunk):
        """Callback when audio chunk is ready."""
        if chunk.is_speech:
            self._chunk_queue.put(chunk)
    
    async def _process_chunk(self, chunk: AudioChunk):
        """Process a single audio chunk through the pipeline."""
        try:
            # Run diarization and ASR concurrently
            diarization_task = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._diarizer.process_with_fallback(
                    chunk.data, 
                    chunk.sample_rate, 
                    chunk.start_time
                )
            )
            
            asr_task = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._asr.transcribe(
                    chunk.data, 
                    chunk.sample_rate, 
                    chunk.start_time
                )
            )
            
            # Wait for both to complete
            speaker_segments, transcript_segments = await asyncio.gather(
                diarization_task, asr_task
            )
            
            if not transcript_segments:
                return
            
            # Align speakers with transcripts
            aligned_segments = self._alignment.align_and_merge(
                transcript_segments, speaker_segments
            )
            
            # Translate each segment
            for segment in aligned_segments:
                if not segment.text.strip():
                    continue
                
                result = await self._translator.translate(
                    segment.text,
                    self.source_language,
                    self.target_language,
                )
                
                # Emit translation event
                event = TranslationEvent(
                    speaker_id=segment.speaker_id,
                    speaker_index=segment.speaker_index,
                    original_text=segment.text,
                    translated_text=result.translated_text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                )
                
                self._emit_translation(event)
            
            self._chunks_processed += 1
            
        except Exception as e:
            self._emit_error(f"Chunk processing failed: {e}")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        # Create event loop for this thread
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        
        self._emit_status("Processing started")
        
        while self._is_running:
            try:
                # Get chunk from queue with timeout
                chunk = self._chunk_queue.get(timeout=0.5)
                
                # Process the chunk
                self._event_loop.run_until_complete(
                    self._process_chunk(chunk)
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                self._emit_error(f"Processing loop error: {e}")
        
        # Cleanup
        self._event_loop.close()
        self._emit_status("Processing stopped")
    
    def _audio_frame_callback(self, frame: AudioFrame):
        """Handle incoming audio frames."""
        chunks = self._audio_buffer.add_frame(frame.data, frame.timestamp)
        for chunk in chunks:
            self._on_audio_chunk(chunk)
    
    def start(self):
        """Start the translation pipeline."""
        if self._is_running:
            return
        
        self._is_running = True
        self._start_time = time.time()
        self._chunks_processed = 0
        
        # Clear queues
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset modules
        self._audio_buffer.reset()
        if hasattr(self._diarizer, 'reset_speaker_mapping'):
            self._diarizer.reset_speaker_mapping()
        
        # Set up buffer callback
        self._audio_buffer.on_chunk = self._on_audio_chunk
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()
        
        # Start audio capture with frame callback
        self._audio_capture.on_frame = self._audio_frame_callback
        self._audio_capture.start()
        
        self._emit_status("Pipeline started - listening...")
        print(f"[Pipeline] Started (src={self.source_language}, tgt={self.target_language})")
    
    def stop(self):
        """Stop the translation pipeline."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop audio capture
        self._audio_capture.stop()
        
        # Flush remaining audio
        final_chunk = self._audio_buffer.flush()
        if final_chunk and final_chunk.is_speech:
            self._on_audio_chunk(final_chunk)
        
        # Wait for processing thread
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
            self._processing_thread = None
        
        self._emit_status("Pipeline stopped")
        print(f"[Pipeline] Stopped (processed {self._chunks_processed} chunks)")
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "is_running": self._is_running,
            "chunks_processed": self._chunks_processed,
            "elapsed_seconds": elapsed,
            "queue_size": self._chunk_queue.qsize(),
        }
    
    def update_languages(self, source: str, target: str):
        """Update source and target languages."""
        self.source_language = source
        self.target_language = target
        
        # Update ASR language if not mock
        if hasattr(self._asr, 'language'):
            self._asr.language = source
        
        print(f"[Pipeline] Languages updated: {source} → {target}")
    
    async def close(self):
        """Clean up resources."""
        self.stop()
        if hasattr(self._translator, 'close'):
            await self._translator.close()
