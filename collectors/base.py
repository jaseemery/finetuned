"""Base collector class with common functionality."""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from diskcache import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CollectedDocument:
    """Represents a collected document."""

    id: str
    source: str  # 'docs', 'github', 'stackoverflow', etc.
    url: str
    title: str
    content: str
    content_type: str  # 'documentation', 'code', 'tutorial', 'qa', etc.
    language: Optional[str] = None  # Programming language if code
    metadata: dict = field(default_factory=dict)
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CollectedDocument":
        return cls(**data)

    def content_hash(self) -> str:
        """Generate a hash of the content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()


class BaseCollector(ABC):
    """Base class for data collectors."""

    def __init__(
        self,
        cache_dir: Path,
        requests_per_second: float = 2.0,
        max_concurrent: int = 5,
        timeout: int = 30,
    ):
        self.cache = Cache(str(cache_dir))
        self.rate_limiter = AsyncLimiter(requests_per_second, 1)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_url(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetch a URL with rate limiting and caching."""
        cache_key = f"url:{url}"

        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit: {url}")
            return self.cache[cache_key]

        async with self.semaphore:
            await self.rate_limiter.acquire()
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if use_cache:
                            self.cache.set(cache_key, content, expire=86400)  # 24h cache
                        return content
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    @abstractmethod
    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """Collect documents. Must be implemented by subclasses."""
        pass

    def save_documents(self, documents: list[CollectedDocument], output_path: Path):
        """Save collected documents to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(documents)} documents to {output_path}")

    def load_documents(self, input_path: Path) -> list[CollectedDocument]:
        """Load documents from a JSONL file."""
        documents = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(CollectedDocument.from_dict(json.loads(line)))
        return documents
