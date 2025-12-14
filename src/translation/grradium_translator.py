"""
Grradium Translation Module

Text translation using the Grradium API.
"""
import asyncio
import httpx
from typing import Optional, Dict
from dataclasses import dataclass
import time
import hashlib


@dataclass
class TranslationResult:
    """Container for translation result."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    success: bool
    error_message: Optional[str] = None
    cached: bool = False


class GrradiumTranslator:
    """
    Translation using Grradium API.
    
    Provides text-to-text translation with caching and retry logic.
    
    Usage:
        translator = GrradiumTranslator(api_key="...")
        result = await translator.translate("Hello, world!", "en", "es")
        print(result.translated_text)  # "Hola, mundo!"
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: str = "https://api.grradium.com/v1/translate",
        timeout: float = 10.0,
        max_retries: int = 1,
        cache_enabled: bool = True,
    ):
        """
        Initialize Grradium translator.
        
        Args:
            api_key: Grradium API key
            api_endpoint: API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            cache_enabled: Whether to cache translation results
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        
        self._cache: Dict[str, str] = {}
        self._client: Optional[httpx.AsyncClient] = None
        
    def _get_cache_key(self, text: str, source: str, target: str) -> str:
        """Generate cache key for translation."""
        content = f"{source}:{target}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., "en")
            target_language: Target language code (e.g., "es")
        
        Returns:
            TranslationResult with translated text or error
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                success=True,
            )
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(text, source_language, target_language)
            if cache_key in self._cache:
                return TranslationResult(
                    original_text=text,
                    translated_text=self._cache[cache_key],
                    source_language=source_language,
                    target_language=target_language,
                    success=True,
                    cached=True,
                )
        
        # Make API request with retry
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._make_request(text, source_language, target_language)
                
                # Cache successful result
                if self.cache_enabled and result.success:
                    self._cache[cache_key] = result.translated_text
                
                return result
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5)  # Brief delay before retry
        
        # All retries failed
        return TranslationResult(
            original_text=text,
            translated_text=text,  # Fallback to original
            source_language=source_language,
            target_language=target_language,
            success=False,
            error_message=f"Translation failed: {last_error}",
        )
    
    async def _make_request(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Make the actual API request."""
        client = await self._get_client()
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        
        response = await client.post(
            self.api_endpoint,
            json=payload,
            headers=headers,
        )
        
        if response.status_code != 200:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                success=False,
                error_message=f"API error: {response.status_code}",
            )
        
        data = response.json()
        translated_text = data.get("translated_text") or data.get("translation") or text
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            success=True,
        )
    
    def translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """
        Synchronous translation wrapper.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
        
        Returns:
            TranslationResult
        """
        return asyncio.run(self.translate(text, source_language, target_language))
    
    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts concurrently.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
        
        Returns:
            List of TranslationResult objects
        """
        tasks = [
            self.translate(text, source_language, target_language)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    def clear_cache(self):
        """Clear translation cache."""
        self._cache = {}
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class MockTranslator:
    """
    Mock translator for testing without API access.
    
    Returns placeholder translations.
    """
    
    def __init__(self, **kwargs):
        self._translations = {
            "en-es": {
                "hello": "hola",
                "how are you": "cómo estás",
                "goodbye": "adiós",
            }
        }
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Return mock translation."""
        # Simple mock: prefix with [TRANSLATED]
        translated = f"[{target_language.upper()}] {text}"
        
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            success=True,
        )
    
    def translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        return asyncio.run(self.translate(text, source_language, target_language))
    
    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[TranslationResult]:
        tasks = [self.translate(t, source_language, target_language) for t in texts]
        return await asyncio.gather(*tasks)
    
    def clear_cache(self):
        pass
    
    async def close(self):
        pass
