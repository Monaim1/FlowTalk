import asyncio
import gradium

async def main():
    client = gradium.client.GradiumClient(api_key="your-api-key")

    # Audio generator that yields audio chunks
    async def audio_generator(audio_data, chunk_size=1920):
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i : i + chunk_size]

    # Create STT stream
    stream = await client.stt_stream(
        {"model_name": "default", "input_format": "pcm"},
        audio_generator(audio_data),
    )

    # Process transcription results
    async for message in stream.iter_text():
        print(message)

if __name__ == "__main__":
    asyncio.run(main())