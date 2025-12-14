"""
Gradio UI for FlowTalk

Real-time speech translation interface with speaker color-coding.
"""
import gradio as gr
from typing import Optional, List
import threading
import time

from src.pipeline.orchestrator import PipelineOrchestrator, TranslationEvent
from config import SUPPORTED_LANGUAGES, SPEAKER_COLORS, config


class TranslationUI:
    """
    Gradio-based web interface for real-time translation.
    
    Features:
    - Language selection (source/target)
    - Start/Stop recording controls
    - Real-time transcript display with speaker colors
    - Toggle for original text visibility
    """
    
    def __init__(
        self,
        pipeline: Optional[PipelineOrchestrator] = None,
        use_mock: bool = False,
    ):
        """
        Initialize the UI.
        
        Args:
            pipeline: Optional pre-configured pipeline
            use_mock: Use mock modules for testing
        """
        self.pipeline = pipeline
        self.use_mock = use_mock
        self._transcript_history: List[TranslationEvent] = []
        self._is_recording = False
        self._lock = threading.Lock()
    
    def _format_speaker_html(self, speaker_id: str, speaker_index: int) -> str:
        """Format speaker label with color."""
        color = SPEAKER_COLORS[speaker_index % len(SPEAKER_COLORS)]
        return f'<span style="color: {color}; font-weight: bold;">[{speaker_id}]</span>'
    
    def _format_transcript_html(self, show_original: bool = True) -> str:
        """Generate HTML for transcript display."""
        if not self._transcript_history:
            return '<div style="color: #888; font-style: italic;">Waiting for speech...</div>'
        
        html_parts = []
        for event in self._transcript_history:
            speaker_html = self._format_speaker_html(event.speaker_id, event.speaker_index)
            timestamp = event.format_timestamp()
            
            if show_original:
                html_parts.append(f'''
                <div style="margin-bottom: 16px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="margin-bottom: 8px;">
                        {speaker_html} <span style="color: #888; font-size: 0.85em;">{timestamp}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #666; font-size: 0.9em;">Original:</span>
                        <span style="color: #333;">{event.original_text}</span>
                    </div>
                    <div>
                        <span style="color: #666; font-size: 0.9em;">Translation:</span>
                        <span style="color: #1a73e8; font-weight: 500;">{event.translated_text}</span>
                    </div>
                </div>
                ''')
            else:
                html_parts.append(f'''
                <div style="margin-bottom: 12px; padding: 10px; background: #f0f7ff; border-radius: 8px;">
                    {speaker_html} <span style="color: #888; font-size: 0.85em;">{timestamp}</span>
                    <div style="color: #1a73e8; margin-top: 4px;">{event.translated_text}</div>
                </div>
                ''')
        
        return ''.join(html_parts)
    
    def _on_translation(self, event: TranslationEvent):
        """Handle new translation event."""
        with self._lock:
            self._transcript_history.append(event)
            # Keep last 50 entries
            if len(self._transcript_history) > 50:
                self._transcript_history = self._transcript_history[-50:]
    
    def _start_recording(
        self, 
        source_lang: str, 
        target_lang: str,
    ) -> tuple:
        """Start recording and translation."""
        if self._is_recording:
            return "Already recording...", self._format_transcript_html(True)
        
        # Get language codes from display names
        source_code = None
        target_code = None
        for code, name in SUPPORTED_LANGUAGES.items():
            if name == source_lang:
                source_code = code
            if name == target_lang:
                target_code = code
        
        if not source_code or not target_code:
            return "Error: Invalid language selection", ""
        
        # Clear history
        with self._lock:
            self._transcript_history = []
        
        # Initialize pipeline if needed
        if self.pipeline is None:
            self.pipeline = PipelineOrchestrator(
                source_language=source_code,
                target_language=target_code,
                use_mock=self.use_mock,
                hf_token=config.diarization.hf_token,
                grradium_api_key=config.translation.api_key,
                grradium_api_endpoint=config.translation.api_endpoint,
                whisper_model=config.asr.model_name,
            )
        else:
            self.pipeline.update_languages(source_code, target_code)
        
        # Set callback
        self.pipeline.on_translation = self._on_translation
        
        # Start pipeline
        self.pipeline.start()
        self._is_recording = True
        
        return f"🎤 Recording... ({source_lang} → {target_lang})", ""
    
    def _stop_recording(self) -> tuple:
        """Stop recording."""
        if not self._is_recording:
            return "Not recording", self._format_transcript_html(True)
        
        if self.pipeline:
            self.pipeline.stop()
        
        self._is_recording = False
        
        return "⏹️ Stopped", self._format_transcript_html(True)
    
    def _refresh_transcript(self, show_original: bool) -> str:
        """Refresh transcript display."""
        return self._format_transcript_html(show_original)
    
    def create_app(self) -> gr.Blocks:
        """Create the Gradio app interface."""
        
        # Custom CSS for styling
        custom_css = """
        .transcript-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 16px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .status-box {
            padding: 12px;
            border-radius: 8px;
            background: #e8f5e9;
            text-align: center;
            font-weight: 500;
        }
        """
        
        with gr.Blocks(
            title="FlowTalk - Real-Time Translator",
            css=custom_css,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="green",
            )
        ) as app:
            
            # Header
            gr.Markdown("""
            # 🌐 FlowTalk
            ### Real-Time Spoken Language Translator
            
            Speak into your microphone and see live translations with speaker identification.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Language selection
                    source_lang = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.values()),
                        value="English",
                        label="Source Language",
                        interactive=True,
                    )
                    target_lang = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.values()),
                        value="Spanish",
                        label="Target Language",
                        interactive=True,
                    )
                    
                    # Controls
                    with gr.Row():
                        start_btn = gr.Button("🎤 Start Recording", variant="primary", size="lg")
                        stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="lg")
                    
                    # Options
                    show_original = gr.Checkbox(
                        label="Show original text",
                        value=True,
                        interactive=True,
                    )
                    
                    # Status display
                    status = gr.Textbox(
                        label="Status",
                        value="Ready to record",
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                
                with gr.Column(scale=2):
                    # Transcript display
                    transcript = gr.HTML(
                        value='<div style="color: #888; font-style: italic; padding: 20px;">Click "Start Recording" to begin...</div>',
                        label="Live Transcript",
                        elem_classes=["transcript-container"],
                    )
                    
                    # Refresh button for manual update
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            # Event handlers
            start_btn.click(
                fn=self._start_recording,
                inputs=[source_lang, target_lang],
                outputs=[status, transcript],
            )
            
            stop_btn.click(
                fn=self._stop_recording,
                inputs=[],
                outputs=[status, transcript],
            )
            
            refresh_btn.click(
                fn=self._refresh_transcript,
                inputs=[show_original],
                outputs=[transcript],
            )
            
            show_original.change(
                fn=self._refresh_transcript,
                inputs=[show_original],
                outputs=[transcript],
            )
            
            # Auto-refresh using a timer (every 2 seconds)
            app.load(
                fn=lambda: self._refresh_transcript(True),
                outputs=[transcript],
                every=2,
            )
            
            # Footer
            gr.Markdown("""
            ---
            *FlowTalk uses PyAnnote for speaker diarization, Whisper for speech recognition, and Grradium for translation.*
            """)
        
        return app


def create_app(use_mock: bool = False) -> gr.Blocks:
    """
    Create and return the Gradio app.
    
    Args:
        use_mock: Use mock modules for testing
    
    Returns:
        Gradio Blocks app
    """
    ui = TranslationUI(use_mock=use_mock)
    return ui.create_app()


def launch_app(
    use_mock: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    """
    Launch the Gradio app.
    
    Args:
        use_mock: Use mock modules for testing
        server_name: Server hostname
        server_port: Server port
        share: Create public URL
    """
    app = create_app(use_mock=use_mock)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    launch_app(use_mock=True)
