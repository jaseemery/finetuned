"""Collector for Temporal.io GitHub repositories."""

import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp
from dotenv import load_dotenv

from .base import BaseCollector, CollectedDocument
from config import Config, TEMPORAL_CONCEPTS

load_dotenv()
logger = logging.getLogger(__name__)


class GitHubCollector(BaseCollector):
    """Collector for code and documentation from GitHub repositories."""

    def __init__(self, config: Config, github_token: Optional[str] = None):
        super().__init__(
            cache_dir=config.cache_dir / "github",
            requests_per_second=config.github_requests_per_second,
            max_concurrent=config.max_concurrent_requests,
            timeout=config.request_timeout,
        )
        self.config = config
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        self.api_base = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TemporalDataCollector/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        else:
            logger.warning(
                "No GitHub token provided. Rate limits will be strict (60 req/hour). "
                "Set GITHUB_TOKEN environment variable for higher limits."
            )

    async def fetch_api(self, endpoint: str, use_cache: bool = True) -> Optional[dict]:
        """Fetch from GitHub API with rate limiting and caching."""
        url = f"{self.api_base}{endpoint}" if endpoint.startswith("/") else endpoint
        cache_key = f"github:{endpoint}"

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        async with self.semaphore:
            await self.rate_limiter.acquire()
            try:
                async with self.session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if use_cache:
                            self.cache.set(cache_key, data, expire=3600)  # 1h cache
                        return data
                    elif response.status == 403:
                        logger.error("GitHub API rate limit exceeded")
                        return None
                    else:
                        logger.warning(f"GitHub API {response.status} for {endpoint}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {endpoint}: {e}")
                return None

    async def get_repo_contents(
        self, owner: str, repo: str, path: str = ""
    ) -> list[dict]:
        """Get contents of a repository directory."""
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        contents = await self.fetch_api(endpoint)
        return contents if contents else []

    async def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get the content of a single file."""
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        data = await self.fetch_api(endpoint)

        if not data or "content" not in data:
            return None

        try:
            content = base64.b64decode(data["content"]).decode("utf-8")
            return content
        except Exception as e:
            logger.error(f"Error decoding file {path}: {e}")
            return None

    async def search_code(
        self, query: str, repo: Optional[str] = None, language: Optional[str] = None
    ) -> list[dict]:
        """Search for code in repositories."""
        search_query = query
        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"

        # Add Temporal org filter
        if "repo:" not in search_query:
            search_query += " org:temporalio"

        endpoint = f"/search/code?q={search_query}&per_page=100"
        data = await self.fetch_api(endpoint, use_cache=False)

        return data.get("items", []) if data else []

    def _detect_language(self, path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".js": "javascript",
            ".go": "go",
            ".java": "java",
            ".md": "markdown",
            ".mdx": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
        }
        ext = Path(path).suffix.lower()
        return ext_map.get(ext)

    def _classify_file(self, path: str, content: str) -> str:
        """Classify file content type."""
        path_lower = path.lower()

        if "test" in path_lower or "_test" in path_lower:
            return "test"
        elif "sample" in path_lower or "example" in path_lower:
            return "example"
        elif path.endswith((".md", ".mdx")):
            return "documentation"
        elif "readme" in path_lower:
            return "documentation"
        elif any(pattern in path_lower for pattern in ["workflow", "activity", "worker"]):
            return "implementation"
        else:
            return "code"

    def _is_relevant_file(self, path: str, content: str) -> bool:
        """Check if file content is relevant to Temporal concepts."""
        content_lower = content.lower()
        path_lower = path.lower()

        # Check for Temporal-specific patterns
        temporal_patterns = [
            r"@workflow",
            r"@activity",
            r"workflow\.defn",
            r"activity\.defn",
            r"temporal\.",
            r"temporalio",
            r"import.*temporal",
            r"from.*temporal",
            r"workflow\.run",
            r"start_workflow",
            r"execute_workflow",
            r"execute_activity",
            r"task_queue",
            r"taskqueue",
        ]

        for pattern in temporal_patterns:
            if re.search(pattern, content_lower):
                return True

        # Check if path suggests Temporal-related content
        temporal_path_patterns = ["temporal", "workflow", "activity", "worker"]
        for pattern in temporal_path_patterns:
            if pattern in path_lower:
                return True

        return False

    async def collect_repo_files(
        self, owner: str, repo: str, path: str = ""
    ) -> AsyncIterator[CollectedDocument]:
        """Recursively collect relevant files from a repository."""
        contents = await self.get_repo_contents(owner, repo, path)

        for item in contents:
            if item["type"] == "dir":
                # Skip certain directories
                skip_dirs = ["node_modules", "vendor", ".git", "__pycache__", "dist", "build"]
                if item["name"] not in skip_dirs:
                    async for doc in self.collect_repo_files(owner, repo, item["path"]):
                        yield doc

            elif item["type"] == "file":
                # Check file extension
                ext = Path(item["name"]).suffix.lower()
                if ext not in self.config.code_extensions:
                    continue

                # Get file content
                content = await self.get_file_content(owner, repo, item["path"])
                if not content:
                    continue

                # Check content length
                if len(content) < self.config.min_content_length:
                    continue
                if len(content) > self.config.max_content_length:
                    content = content[: self.config.max_content_length]

                # Check relevance
                if not self._is_relevant_file(item["path"], content):
                    continue

                language = self._detect_language(item["path"])
                content_type = self._classify_file(item["path"], content)

                doc = CollectedDocument(
                    id=f"github:{owner}/{repo}/{item['path']}",
                    source="github",
                    url=item["html_url"],
                    title=f"{repo}/{item['path']}",
                    content=content,
                    content_type=content_type,
                    language=language,
                    metadata={
                        "repo": f"{owner}/{repo}",
                        "path": item["path"],
                        "sha": item.get("sha"),
                        "size": item.get("size"),
                    },
                )
                yield doc

    async def collect_by_search(self) -> AsyncIterator[CollectedDocument]:
        """Collect files by searching for Temporal concepts."""
        seen_urls = set()

        for concept in TEMPORAL_CONCEPTS[:10]:  # Limit to avoid rate limits
            logger.info(f"Searching GitHub for: {concept}")
            results = await self.search_code(concept)

            for item in results:
                if item["html_url"] in seen_urls:
                    continue
                seen_urls.add(item["html_url"])

                # Parse repo info
                repo_full = item["repository"]["full_name"]
                owner, repo = repo_full.split("/")
                path = item["path"]

                # Get full file content
                content = await self.get_file_content(owner, repo, path)
                if not content or len(content) < self.config.min_content_length:
                    continue

                language = self._detect_language(path)
                content_type = self._classify_file(path, content)

                doc = CollectedDocument(
                    id=f"github:{repo_full}/{path}",
                    source="github",
                    url=item["html_url"],
                    title=f"{repo}/{path}",
                    content=content,
                    content_type=content_type,
                    language=language,
                    metadata={
                        "repo": repo_full,
                        "path": path,
                        "search_term": concept,
                    },
                )
                yield doc

            # Small delay between searches to avoid rate limits
            await asyncio.sleep(2)

    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """Collect from all configured repositories."""
        for repo_full in self.config.github_repos:
            owner, repo = repo_full.split("/")
            logger.info(f"Collecting from {repo_full}")

            try:
                async for doc in self.collect_repo_files(owner, repo):
                    yield doc
            except Exception as e:
                logger.error(f"Error collecting from {repo_full}: {e}")
                continue

        # Also collect via search for broader coverage
        logger.info("Performing concept-based search...")
        async for doc in self.collect_by_search():
            yield doc


async def main():
    """Test the GitHub collector."""
    config = Config()

    async with GitHubCollector(config) as collector:
        documents = []
        async for doc in collector.collect():
            documents.append(doc)
            logger.info(f"Collected: {doc.title} ({doc.content_type}, {doc.language})")

            # Limit for testing
            if len(documents) >= 50:
                break

        collector.save_documents(documents, config.raw_dir / "github_code.jsonl")
        print(f"\nCollected {len(documents)} code files")


if __name__ == "__main__":
    asyncio.run(main())
