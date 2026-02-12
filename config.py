"""Configuration for the Temporal.io data collection pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Pipeline configuration."""

    # Output directories
    output_dir: Path = Path("./data")
    raw_dir: Path = field(default_factory=lambda: Path("./data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("./data/processed"))
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))

    # Temporal documentation URLs
    docs_base_url: str = "https://docs.temporal.io"
    docs_sitemap_url: str = "https://docs.temporal.io/sitemap.xml"

    # GitHub repositories to scrape
    github_repos: list = field(default_factory=lambda: [
        "temporalio/temporal",
        "temporalio/sdk-python",
        "temporalio/sdk-typescript",
        "temporalio/sdk-go",
        "temporalio/sdk-java",
        "temporalio/samples-python",
        "temporalio/samples-typescript",
        "temporalio/samples-go",
        "temporalio/samples-java",
        "temporalio/documentation",
    ])

    # File extensions to collect from GitHub
    code_extensions: list = field(default_factory=lambda: [
        ".py", ".ts", ".js", ".go", ".java", ".md", ".mdx"
    ])

    # Rate limiting
    requests_per_second: float = 2.0
    github_requests_per_second: float = 1.0

    # Scraping settings
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    max_retries: int = 3

    # Content settings
    min_content_length: int = 100  # Minimum characters for useful content
    max_content_length: int = 50000  # Maximum characters per document

    def __post_init__(self):
        """Create output directories."""
        for directory in [self.output_dir, self.raw_dir, self.processed_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Temporal concepts for focused data collection
TEMPORAL_CONCEPTS = [
    "workflow",
    "activity",
    "worker",
    "task queue",
    "signal",
    "query",
    "timer",
    "child workflow",
    "continue as new",
    "retry policy",
    "timeout",
    "schedule",
    "namespace",
    "search attributes",
    "visibility",
    "event history",
    "replay",
    "determinism",
    "saga pattern",
    "compensation",
    "idempotency",
    "workflow execution",
    "workflow definition",
    "temporal client",
    "temporal server",
    "temporal cloud",
]

# Documentation sections to prioritize
PRIORITY_DOC_PATHS = [
    "/workflows",
    "/activities",
    "/workers",
    "/tasks",
    "/encyclopedia",
    "/develop",
    "/production-deployment",
    "/references",
    "/glossary",
]
