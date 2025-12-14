import asyncio
import os
import logging
import sys
import datetime
import numpy as np
import sounddevice as sd
from transformers import MarianMTModel, MarianTokenizer
import gradium
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.services.audio_capture import microphone_stream

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

# Create records directory
RECORDS_DIR = os.path.join(os.path.dirname(__file__), '../../records')
os.makedirs(RECORDS_DIR, exist_ok=True)

class SessionRecorder:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(RECORDS_DIR, f"session_{timestamp}.txt")
        with open(self.filename, 'w') as f:
            f.write(f"Translation Session Started: {timestamp}\n")
            f.write("----------------------------------------\n\n")
        logger.info(f"Recording session to: {self.filename}")

    def log(self, transcription, translation):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            with open(self.filename, 'a') as f:
                f.write(f"[{timestamp}] [Transcription]: {transcription}\n")
                f.write(f"[{timestamp}] [Translation]:   {translation}\n")
                f.write("-" * 40 + "\n")
        except Exception as e:
            logger.error(f"Failed to write to record file: {e}")

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

async def process_buffer_and_translate(queue, translator, client, recorder):
    buffer = []
    LAG_SECONDS = 0.5
    
    # Get Voice ID from env or default
    voice_id = os.getenv("TTS_VOICE_ID", "wzsx4FjdulIY7oBs")
    
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
                
                translated_text = translator.translate(full_text)
                
                print(f"\n[Transcription]: {full_text}")
                print(f"[Translation]:   {translated_text}")
                print("-" * 40)
                
                # Record to file
                recorder.log(full_text, translated_text)
                
                # TTS
                try:
                    tts_result = await client.tts(
                         setup={
                             "model_name": "default",
                             "voice_id": voice_id,
                             "output_format": "pcm"
                         },
                         text=translated_text
                    )
                    
                    if hasattr(tts_result, 'pcm16'):
                        audio_data = tts_result.pcm16()
                    else:
                        audio_data = np.frombuffer(tts_result.raw_data, dtype=np.int16)

                    # Play audio
                    sd.play(audio_data, samplerate=48000)
                    sd.wait()
                    
                except Exception as tts_error:
                   logger.error(f"TTS Error: {tts_error}")
                
                buffer = []
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")

async def main():
    # 1. Initialize Services
    translator = TranslationService()
    recorder = SessionRecorder()

    # 2. Initialize Gradium Clients
    api_key = os.getenv("GRADIUM_API_KEY")
    if not api_key:
        logger.warning("GRADIUM_API_KEY environment variable not set. Using placeholder 'your-api-key'.")
        api_key = "your-api-key"
    
    client = gradium.client.GradiumClient(api_key=api_key)
    tts_client = gradium.client.GradiumClient(api_key=api_key)
    logger.info("Gradium clients initialized.")

    # 3. Start Pipeline
    logger.info("Starting pipeline. Speak into the microphone (French)...")

    try:
        stream = await client.stt_stream(
            {"model_name": "default", "input_format": "pcm"},
            microphone_stream()
        )

        logger.info("STT Stream connected. Waiting for transcription...")

        queue = asyncio.Queue()
        consumer_task = asyncio.create_task(process_buffer_and_translate(queue, translator, tts_client, recorder))

        async for text_segment in stream.iter_text():
            if text_segment:
                text_content = text_segment.text if hasattr(text_segment, 'text') else str(text_segment)
                if text_content.strip():
                    await queue.put(text_content)
        
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        logger.error(f"Pipeline error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.")
