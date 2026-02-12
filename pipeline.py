"""Main data collection pipeline for Temporal.io fine-tuning data."""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from config import Config
from collectors import TemporalDocsScraper, GitHubCollector
from collectors.base import CollectedDocument
from processors import QAGenerator, DataFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TemporalDataPipeline:
    """Main pipeline for collecting and processing Temporal.io training data."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.stats = {
            "docs_collected": 0,
            "github_collected": 0,
            "qa_pairs_generated": 0,
            "start_time": None,
            "end_time": None,
        }

    async def collect_documentation(self) -> list[CollectedDocument]:
        """Collect Temporal documentation."""
        logger.info("=" * 50)
        logger.info("STEP 1: Collecting Temporal Documentation")
        logger.info("=" * 50)

        documents = []
        async with TemporalDocsScraper(self.config) as scraper:
            async for doc in scraper.collect():
                documents.append(doc)
                logger.info(f"  Collected: {doc.title[:50]}...")

        # Save raw documents
        output_path = self.config.raw_dir / "temporal_docs.jsonl"
        self._save_documents(documents, output_path)

        self.stats["docs_collected"] = len(documents)
        logger.info(f"Collected {len(documents)} documentation pages")
        return documents

    async def collect_github_code(
        self, limit: int = None
    ) -> list[CollectedDocument]:
        """Collect code from GitHub repositories."""
        logger.info("=" * 50)
        logger.info("STEP 2: Collecting GitHub Code Examples")
        logger.info("=" * 50)

        documents = []
        async with GitHubCollector(self.config) as collector:
            async for doc in collector.collect():
                documents.append(doc)
                logger.info(f"  Collected: {doc.title[:50]}...")

                if limit and len(documents) >= limit:
                    break

        # Save raw documents
        output_path = self.config.raw_dir / "github_code.jsonl"
        self._save_documents(documents, output_path)

        self.stats["github_collected"] = len(documents)
        logger.info(f"Collected {len(documents)} code files")
        return documents

    async def generate_qa_pairs(
        self,
        documents: list[CollectedDocument],
        use_llm: bool = False,
    ) -> list:
        """Generate Q&A pairs from collected documents."""
        logger.info("=" * 50)
        logger.info("STEP 3: Generating Q&A Pairs")
        logger.info("=" * 50)

        generator = QAGenerator(use_llm=use_llm)
        qa_pairs = await generator.process_documents(
            documents, self.config.processed_dir / "qa_pairs.jsonl"
        )

        self.stats["qa_pairs_generated"] = len(qa_pairs)
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs

    def format_for_training(
        self,
        qa_pairs: list,
        formats: list[str] = None,
        template: str = "llama3",
    ) -> dict:
        """Format Q&A pairs for fine-tuning."""
        logger.info("=" * 50)
        logger.info("STEP 4: Formatting for Training")
        logger.info("=" * 50)

        formatter = DataFormatter(template_name=template)
        output_paths = formatter.process_and_save(
            qa_pairs,
            self.config.processed_dir / "training_data",
            formats=formats or ["trl", "torchtune", "alpaca"],
        )

        return output_paths

    async def run(
        self,
        skip_docs: bool = False,
        skip_github: bool = False,
        github_limit: int = None,
        use_llm: bool = False,
        formats: list[str] = None,
        template: str = "llama3",
    ) -> dict:
        """
        Run the complete pipeline.

        Args:
            skip_docs: Skip documentation collection
            skip_github: Skip GitHub collection
            github_limit: Limit number of GitHub files to collect
            use_llm: Use LLM for Q&A generation
            formats: Output formats for training data
            template: Chat template name

        Returns:
            Pipeline statistics and output paths
        """
        self.stats["start_time"] = datetime.utcnow().isoformat()
        all_documents = []

        # Step 1: Collect documentation
        if not skip_docs:
            docs = await self.collect_documentation()
            all_documents.extend(docs)
        else:
            # Try to load existing docs
            docs_path = self.config.raw_dir / "temporal_docs.jsonl"
            if docs_path.exists():
                all_documents.extend(self._load_documents(docs_path))
                logger.info(f"Loaded {len(all_documents)} existing documentation pages")

        # Step 2: Collect GitHub code
        if not skip_github:
            code = await self.collect_github_code(limit=github_limit)
            all_documents.extend(code)
        else:
            # Try to load existing code
            code_path = self.config.raw_dir / "github_code.jsonl"
            if code_path.exists():
                code_docs = self._load_documents(code_path)
                all_documents.extend(code_docs)
                logger.info(f"Loaded {len(code_docs)} existing code files")

        logger.info(f"\nTotal documents collected: {len(all_documents)}")

        # Step 3: Generate Q&A pairs
        qa_pairs = await self.generate_qa_pairs(all_documents, use_llm=use_llm)

        # Step 4: Format for training
        output_paths = self.format_for_training(qa_pairs, formats=formats, template=template)

        self.stats["end_time"] = datetime.utcnow().isoformat()

        # Print summary
        self._print_summary(output_paths)

        return {
            "stats": self.stats,
            "output_paths": output_paths,
        }

    def _save_documents(self, documents: list[CollectedDocument], path: Path):
        """Save documents to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    def _load_documents(self, path: Path) -> list[CollectedDocument]:
        """Load documents from JSONL file."""
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(CollectedDocument.from_dict(json.loads(line)))
        return documents

    def _print_summary(self, output_paths: dict):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  Documentation pages: {self.stats['docs_collected']}")
        print(f"  GitHub code files:   {self.stats['github_collected']}")
        print(f"  Q&A pairs generated: {self.stats['qa_pairs_generated']}")
        print(f"\nOutput Files:")
        for fmt, paths in output_paths.items():
            print(f"\n  {fmt}:")
            print(f"    Train:      {paths['train']} ({paths['train_count']} examples)")
            print(f"    Validation: {paths['validation']} ({paths['val_count']} examples)")
        print("\n" + "=" * 60)


def main():
    """Run the pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Collect and process Temporal.io data for LLM fine-tuning"
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip documentation collection (use cached data)",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub collection (use cached data)",
    )
    parser.add_argument(
        "--github-limit",
        type=int,
        default=None,
        help="Limit number of GitHub files to collect",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for Q&A generation (requires API key)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["trl", "torchtune", "alpaca"],
        help="Output formats (trl, torchtune, alpaca, axolotl, raw)",
    )
    parser.add_argument(
        "--template",
        default="llama3",
        choices=["llama3", "mistral", "chatml", "alpaca"],
        help="Chat template for formatting",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for collected data",
    )

    args = parser.parse_args()

    # Create config
    config = Config()
    config.output_dir = Path(args.output_dir)
    config.raw_dir = config.output_dir / "raw"
    config.processed_dir = config.output_dir / "processed"
    config.cache_dir = config.output_dir / "cache"

    # Run pipeline
    pipeline = TemporalDataPipeline(config)
    result = asyncio.run(
        pipeline.run(
            skip_docs=args.skip_docs,
            skip_github=args.skip_github,
            github_limit=args.github_limit,
            use_llm=args.use_llm,
            formats=args.formats,
            template=args.template,
        )
    )

    # Save final stats
    stats_path = config.output_dir / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
