import asyncio
import os
import logging
import sys

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transformers import MarianMTModel, MarianTokenizer
import gradium
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import numpy as np
import sounddevice as sd
from src.services.audio_capture import microphone_stream

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

class TranslationService:
    def __init__(self):
        logger.info(f"Loading translation model: {MODEL_NAME}...")
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(MODEL_NAME)
        logger.info("Translation model loaded.")

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

async def main():
    # 1. Initialize Translation Service
    translator = TranslationService()

    # 2. Initialize Gradium Client
    # Ensure API Key is set
    api_key = os.getenv("GRADIUM_API_KEY")
    if not api_key:
        logger.warning("GRADIUM_API_KEY environment variable not set. Using placeholder 'your-api-key'.")
        api_key = "your-api-key"
    
    client = gradium.client.GradiumClient(api_key=api_key)
    logger.info("Gradium client initialized.")

    # Create a separate client for TTS to avoid "WebSocket limit exceeded" if it's per-socket
    # If it's per-API-key limitation of 1, this won't help and we'll need half-duplex.
    tts_client = gradium.client.GradiumClient(api_key=api_key)
    logger.info("Gradium TTS client initialized.")

    # 3. Start Pipeline
    logger.info("Starting pipeline. Speak into the microphone (French)...")

    try:
        # Create STT stream
        # Note: gradium_stt.py used {"model_name": "default", "input_format": "pcm"}
        # We pass our async generator `microphone_stream()`
        stream = await client.stt_stream(
            {"model_name": "default", "input_format": "pcm"},
            microphone_stream()
        )

        logger.info("STT Stream connected. Waiting for transcription...")

        # Process results
        # Create a queue for buffering
        queue = asyncio.Queue()
        
        # Start the consumer task
        consumer_task = asyncio.create_task(process_buffer_and_translate(queue, translator, tts_client))

        # Producer loop
        async for text_segment in stream.iter_text():
            if text_segment:
                text_content = text_segment.text if hasattr(text_segment, 'text') else str(text_segment)
                if text_content.strip():
                    await queue.put(text_content)
        
        # Clean up
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        logger.error(f"Pipeline error: {e}")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")

async def process_buffer_and_translate(queue, translator, client):
    buffer = []
    LAG_SECONDS = 0.5
    
    while True:
        try:
            # Wait for the first item
            first_item = await queue.get()
            buffer.append(first_item)
            
            # Keep collecting items until timeout
            while True:
                try:
                    # Wait for next item with timeout
                    next_item = await asyncio.wait_for(queue.get(), timeout=LAG_SECONDS)
                    buffer.append(next_item)
                except asyncio.TimeoutError:
                    # Timeout reached, flush buffer
                    break
            
            # Process buffer
            if buffer:
                full_text = " ".join(buffer)
                
                # logger.info(f"Buffered STT: {full_text}")
                translated_text = translator.translate(full_text)
                
                print(f"\n[Transcription]: {full_text}")
                print(f"[Translation]:   {translated_text}")
                print("-" * 40)
                
                # TTS
                try:
                    # Using "pcm" format as per documentation for streaming/raw usage
                    # Docs say: 48kHz, 16-bit signed integer, mono
                    tts_result = await client.tts(
                         setup={
                             "model_name": "default",
                             "voice_id": "YTpq7expH9539ERJ", # Using default/example voice
                             "output_format": "pcm"
                         },
                         text=translated_text
                    )
                    
                    # specific method mentioned in docs: result.pcm16() or manual
                    # Let's try manual from raw_data if pcm16() isn't standard, 
                    # but docs said `pcm16_array = result.pcm16()` exists.
                    # Safety check:
                    if hasattr(tts_result, 'pcm16'):
                        audio_data = tts_result.pcm16()
                    else:
                        # Fallback based on "16-bit signed integer (little-endian)"
                        audio_data = np.frombuffer(tts_result.raw_data, dtype=np.int16)

                    # Play audio
                    # Sample rate 48000 as per docs
                    sd.play(audio_data, samplerate=48000)
                    sd.wait() # Block this task until audio finishes to prevent overlap logic issues 
                              # (or we could overlap, but simple is better first)
                    
                except Exception as tts_error:
                   logger.error(f"TTS Error: {tts_error}")
                
                buffer = []
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.")
