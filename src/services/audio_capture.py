import sounddevice as sd
import asyncio
import logging
import queue
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio configuration
CHANNELS = 1
RATE = 16000
CHUNK = 1024
DTYPE = 'int16'

async def microphone_stream():
    """
    Async generator that captures audio from the microphone and yields chunks as bytes.
    """
    q = queue.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"Audio status: {status}")
        q.put(bytes(indata))

    try:
        # Open the stream
        # blocksize=CHUNK results in 'frames' in callback being approx CHUNK
        with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, device=None,
                               channels=CHANNELS, dtype=DTYPE,
                               callback=callback):
            logger.info("Microphone stream started. Listening...")
            
            while True:
                # Use run_in_executor to blocking get from queue slightly safer than direct,
                # though queue.get is thread-safe, we don't want to block the async loop.
                # However, for a simple implementation, yielding regularly is key.
                # We can just check the queue non-blockingly or sleep.
                
                # Better approach for async generator:
                while q.empty():
                    await asyncio.sleep(0.01)
                
                data = q.get()
                yield data

    except Exception as e:
        logger.error(f"Error in microphone stream: {e}")
    finally:
        logger.info("Stopping microphone stream...")

if __name__ == "__main__":
    # Test the stream
    async def test():
        print("Recording for 5 seconds...")
        count = 0
        async for chunk in microphone_stream():
            count += 1
            if count > (RATE / CHUNK) * 5:
                # This logic is approximate as chunk size varies, but good enough for test
                break
        print("Done.")

    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        pass

