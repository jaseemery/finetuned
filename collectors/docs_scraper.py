"""Scraper for Temporal.io documentation."""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from .base import BaseCollector, CollectedDocument
from config import Config, PRIORITY_DOC_PATHS

logger = logging.getLogger(__name__)


class TemporalDocsScraper(BaseCollector):
    """Scraper for Temporal.io documentation."""

    def __init__(self, config: Config):
        super().__init__(
            cache_dir=config.cache_dir / "docs",
            requests_per_second=config.requests_per_second,
            max_concurrent=config.max_concurrent_requests,
            timeout=config.request_timeout,
        )
        self.config = config
        self.base_url = config.docs_base_url
        self.sitemap_url = config.docs_sitemap_url
        self.visited_urls: set[str] = set()

    async def get_sitemap_urls(self) -> list[str]:
        """Extract all documentation URLs from the sitemap."""
        content = await self.fetch_url(self.sitemap_url)
        if not content:
            logger.error("Failed to fetch sitemap")
            return []

        urls = []
        try:
            root = ElementTree.fromstring(content)
            # Handle XML namespaces
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for url_elem in root.findall(".//ns:url/ns:loc", namespace):
                url = url_elem.text
                if url and self.is_valid_doc_url(url):
                    urls.append(url)
        except ElementTree.ParseError:
            # Fallback: try regex extraction
            urls = re.findall(r"<loc>(https://docs\.temporal\.io[^<]+)</loc>", content)

        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls

    def is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation page."""
        parsed = urlparse(url)
        path = parsed.path

        # Skip non-documentation pages
        skip_patterns = [
            "/api/",
            "/changelog",
            "/search",
            "/_",
            ".xml",
            ".json",
            "/tags/",
        ]
        for pattern in skip_patterns:
            if pattern in path:
                return False

        return True

    def extract_content(self, html: str, url: str) -> Optional[dict]:
        """Extract structured content from a documentation page."""
        soup = BeautifulSoup(html, "lxml")

        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Try to find the main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|docs|markdown", re.I))
        )

        if not main_content:
            main_content = soup.body

        if not main_content:
            return None

        # Extract title
        title = None
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True).replace(" | Temporal", "")

        # Extract text content
        text_content = self._extract_text(main_content)

        if len(text_content) < self.config.min_content_length:
            return None

        # Extract code blocks
        code_blocks = []
        for code in main_content.find_all("pre"):
            code_text = code.get_text(strip=True)
            if code_text:
                # Try to detect language from class
                lang = None
                code_elem = code.find("code")
                if code_elem and code_elem.get("class"):
                    classes = code_elem.get("class", [])
                    for cls in classes:
                        if cls.startswith("language-"):
                            lang = cls.replace("language-", "")
                            break
                code_blocks.append({"code": code_text, "language": lang})

        # Determine content type based on URL path
        content_type = self._classify_content(url)

        return {
            "title": title or "Untitled",
            "content": text_content,
            "code_blocks": code_blocks,
            "content_type": content_type,
        }

    def _extract_text(self, element) -> str:
        """Extract clean text from HTML element."""
        # Get text with proper spacing
        texts = []
        for string in element.stripped_strings:
            texts.append(string)

        text = " ".join(texts)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _classify_content(self, url: str) -> str:
        """Classify content type based on URL."""
        path = urlparse(url).path.lower()

        if "/develop" in path or "/sdk" in path:
            return "tutorial"
        elif "/references" in path or "/api" in path:
            return "reference"
        elif "/encyclopedia" in path or "/glossary" in path:
            return "concept"
        elif "/production" in path or "/deploy" in path:
            return "guide"
        else:
            return "documentation"

    async def scrape_page(self, url: str) -> Optional[CollectedDocument]:
        """Scrape a single documentation page."""
        if url in self.visited_urls:
            return None

        self.visited_urls.add(url)
        html = await self.fetch_url(url)

        if not html:
            return None

        extracted = self.extract_content(html, url)
        if not extracted:
            return None

        # Create document with code blocks in metadata
        doc = CollectedDocument(
            id=f"docs:{urlparse(url).path}",
            source="temporal_docs",
            url=url,
            title=extracted["title"],
            content=extracted["content"],
            content_type=extracted["content_type"],
            metadata={
                "code_blocks": extracted["code_blocks"],
                "path": urlparse(url).path,
            },
        )

        return doc

    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """Collect all documentation pages."""
        urls = await self.get_sitemap_urls()

        if not urls:
            # Fallback: crawl from known starting points
            urls = [urljoin(self.base_url, path) for path in PRIORITY_DOC_PATHS]

        # Prioritize important documentation sections
        def priority_key(url: str) -> int:
            path = urlparse(url).path
            for i, priority_path in enumerate(PRIORITY_DOC_PATHS):
                if path.startswith(priority_path):
                    return i
            return len(PRIORITY_DOC_PATHS)

        urls = sorted(set(urls), key=priority_key)
        logger.info(f"Scraping {len(urls)} documentation pages")

        # Process URLs concurrently
        tasks = [self.scrape_page(url) for url in urls]
        for coro in asyncio.as_completed(tasks):
            doc = await coro
            if doc:
                yield doc


async def main():
    """Test the documentation scraper."""
    config = Config()

    async with TemporalDocsScraper(config) as scraper:
        documents = []
        async for doc in scraper.collect():
            documents.append(doc)
            logger.info(f"Collected: {doc.title} ({doc.content_type})")

        scraper.save_documents(documents, config.raw_dir / "temporal_docs.jsonl")
        print(f"\nCollected {len(documents)} documentation pages")


if __name__ == "__main__":
    asyncio.run(main())
