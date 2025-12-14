#!/usr/bin/env python3
"""
FlowTalk - Real-Time Spoken Language Translator

Main entry point for the application.
"""
import argparse
import sys
from typing import Optional

from config import config, SUPPORTED_LANGUAGES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FlowTalk - Real-Time Spoken Language Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start with defaults (English → Spanish)
  python main.py --source en --target fr  # English to French
  python main.py --mock                   # Test mode without API keys
  python main.py --share                  # Create public URL for sharing
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Source language code (default: en)"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="es",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Target language code (default: es)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        choices=[
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-medium",
        ],
        help="Whisper model to use (default: whisper-small)"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock modules for testing (no API keys needed)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public URL for sharing"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--console",
        action="store_true",
        help="Run in console mode (no web UI)"
    )
    
    return parser.parse_args()


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    return True


def check_api_keys(use_mock: bool) -> bool:
    """Check if required API keys are configured."""
    if use_mock:
        return True
    
    warnings = []
    
    if not config.diarization.hf_token:
        warnings.append("HF_TOKEN not set - PyAnnote diarization may fail")
    
    if not config.translation.api_key:
        warnings.append("GRRADIUM_API_KEY not set - Translation will fail")
    
    if warnings:
        print("⚠️  API Key Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\nCopy .env.example to .env and add your API keys.")
        print("Or use --mock flag for testing without API keys.\n")
    
    return True


def run_console_mode(args):
    """Run in console mode with colored output."""
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from src.pipeline.orchestrator import PipelineOrchestrator, TranslationEvent
    
    console = Console()
    
    # Update config
    config.source_language = args.source
    config.target_language = args.target
    config.asr.model_name = args.model
    config.asr.use_gpu = not args.no_gpu
    config.diarization.use_gpu = not args.no_gpu
    
    # Create pipeline
    pipeline = PipelineOrchestrator(
        source_language=args.source,
        target_language=args.target,
        use_mock=args.mock,
        use_gpu=not args.no_gpu,
        hf_token=config.diarization.hf_token,
        grradium_api_key=config.translation.api_key,
        grradium_api_endpoint=config.translation.api_endpoint,
        whisper_model=args.model,
    )
    
    # Track translations
    translations = []
    colors = ["blue", "green", "red", "magenta", "cyan", "yellow"]
    
    def on_translation(event: TranslationEvent):
        color = colors[event.speaker_index % len(colors)]
        console.print(
            f"[{color}][{event.speaker_id}][/{color}] "
            f"[dim]{event.format_timestamp()}[/dim]"
        )
        console.print(f"  Original: {event.original_text}")
        console.print(f"  [bold blue]Translation: {event.translated_text}[/bold blue]")
        console.print()
    
    pipeline.on_translation = on_translation
    
    # Print header
    console.print("\n[bold green]🌐 FlowTalk Console Mode[/bold green]")
    console.print(f"   {SUPPORTED_LANGUAGES[args.source]} → {SUPPORTED_LANGUAGES[args.target]}")
    console.print(f"   Model: {args.model}")
    console.print(f"   Mode: {'Mock' if args.mock else 'Live'}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    
    # Run
    try:
        pipeline.start()
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        pipeline.stop()
        console.print("[green]Done![/green]")


def run_web_mode(args):
    """Run in web UI mode."""
    from ui.gradio_app import TranslationUI
    from src.pipeline.orchestrator import PipelineOrchestrator
    
    # Update config
    config.source_language = args.source
    config.target_language = args.target
    config.asr.model_name = args.model
    config.asr.use_gpu = not args.no_gpu
    config.diarization.use_gpu = not args.no_gpu
    
    # Create pipeline
    pipeline = PipelineOrchestrator(
        source_language=args.source,
        target_language=args.target,
        use_mock=args.mock,
        use_gpu=not args.no_gpu,
        hf_token=config.diarization.hf_token,
        grradium_api_key=config.translation.api_key,
        grradium_api_endpoint=config.translation.api_endpoint,
        whisper_model=args.model,
    )
    
    # Create UI
    ui = TranslationUI(pipeline=pipeline, use_mock=args.mock)
    app = ui.create_app()
    
    # Print info
    print("\n🌐 FlowTalk - Real-Time Spoken Language Translator")
    print(f"   {SUPPORTED_LANGUAGES[args.source]} → {SUPPORTED_LANGUAGES[args.target]}")
    print(f"   Model: {args.model}")
    print(f"   Mode: {'Mock' if args.mock else 'Live'}")
    print(f"\n   Opening at http://{args.host}:{args.port}")
    if args.share:
        print("   Public URL will be generated...")
    print()
    
    # Launch
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API keys
    check_api_keys(args.mock)
    
    # Run in appropriate mode
    if args.console:
        run_console_mode(args)
    else:
        run_web_mode(args)


if __name__ == "__main__":
    main()
