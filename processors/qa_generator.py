"""Generate Q&A pairs from collected documents."""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from collectors.base import CollectedDocument
from config import TEMPORAL_CONCEPTS

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a question-answer pair for training."""

    instruction: str
    response: str
    source_id: str
    source_url: str
    category: str  # 'concept', 'howto', 'code', 'troubleshooting'
    concepts: list[str]  # Related Temporal concepts

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "category": self.category,
            "concepts": self.concepts,
        }


class QAGenerator:
    """Generate Q&A pairs from collected documents."""

    def __init__(self, use_llm: bool = False, llm_provider: str = "anthropic"):
        """
        Initialize the Q&A generator.

        Args:
            use_llm: Whether to use an LLM to generate Q&A pairs
            llm_provider: 'anthropic' or 'openai'
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.client = None

        if use_llm:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the LLM client."""
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                logger.warning("anthropic package not installed, falling back to rule-based")
                self.use_llm = False
        elif self.llm_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("openai package not installed, falling back to rule-based")
                self.use_llm = False

    def extract_concepts(self, text: str) -> list[str]:
        """Extract Temporal concepts mentioned in the text."""
        text_lower = text.lower()
        found_concepts = []

        for concept in TEMPORAL_CONCEPTS:
            # Match whole words or hyphenated versions
            pattern = rf"\b{re.escape(concept.replace(' ', '[ -]?'))}\b"
            if re.search(pattern, text_lower):
                found_concepts.append(concept)

        return found_concepts

    def generate_rule_based(self, doc: CollectedDocument) -> list[QAPair]:
        """Generate Q&A pairs using rule-based extraction."""
        qa_pairs = []
        concepts = self.extract_concepts(doc.content)

        if not concepts:
            return qa_pairs

        # Strategy 1: Generate concept explanation questions
        for concept in concepts[:3]:  # Limit per document
            qa = self._generate_concept_qa(doc, concept)
            if qa:
                qa_pairs.append(qa)

        # Strategy 2: Generate how-to questions from code
        if doc.language and doc.content_type in ["example", "code", "implementation"]:
            qa = self._generate_code_qa(doc, concepts)
            if qa:
                qa_pairs.append(qa)

        # Strategy 3: Extract from documentation headers
        if doc.content_type == "documentation":
            qa_pairs.extend(self._extract_from_headers(doc, concepts))

        return qa_pairs

    def _generate_concept_qa(
        self, doc: CollectedDocument, concept: str
    ) -> Optional[QAPair]:
        """Generate a concept explanation Q&A."""
        # Find relevant paragraph about the concept
        paragraphs = doc.content.split("\n\n")
        relevant_para = None

        for para in paragraphs:
            if concept.lower() in para.lower() and len(para) > 100:
                relevant_para = para
                break

        if not relevant_para:
            return None

        # Clean up the paragraph
        response = relevant_para.strip()
        if len(response) > 1500:
            response = response[:1500] + "..."

        question_templates = [
            f"What is a {concept} in Temporal?",
            f"Explain the concept of {concept} in Temporal.",
            f"How does {concept} work in Temporal?",
            f"What is the purpose of {concept} in Temporal?",
        ]

        import random
        question = random.choice(question_templates)

        return QAPair(
            instruction=question,
            response=response,
            source_id=doc.id,
            source_url=doc.url,
            category="concept",
            concepts=[concept],
        )

    def _generate_code_qa(
        self, doc: CollectedDocument, concepts: list[str]
    ) -> Optional[QAPair]:
        """Generate a how-to Q&A from code examples."""
        if not doc.language:
            return None

        # Create a question based on the code content
        lang_name = {
            "python": "Python",
            "typescript": "TypeScript",
            "go": "Go",
            "java": "Java",
        }.get(doc.language, doc.language)

        primary_concept = concepts[0] if concepts else "Temporal"

        question = f"How do I implement a {primary_concept} in Temporal using {lang_name}?"

        # Prepare the response with code
        code_content = doc.content
        if len(code_content) > 2000:
            code_content = code_content[:2000] + "\n# ... (truncated)"

        response = f"Here's an example of implementing a {primary_concept} in {lang_name}:\n\n```{doc.language}\n{code_content}\n```"

        return QAPair(
            instruction=question,
            response=response,
            source_id=doc.id,
            source_url=doc.url,
            category="code",
            concepts=concepts,
        )

    def _extract_from_headers(
        self, doc: CollectedDocument, concepts: list[str]
    ) -> list[QAPair]:
        """Extract Q&A pairs from documentation headers."""
        qa_pairs = []

        # Pattern to find headers and their content
        header_pattern = r"(?:^|\n)(#{1,3})\s+([^\n]+)\n((?:(?!^#{1,3}\s).)*)"
        matches = re.findall(header_pattern, doc.content, re.MULTILINE | re.DOTALL)

        for level, header, content in matches:
            content = content.strip()
            if len(content) < 100:
                continue

            # Skip generic headers
            skip_headers = ["overview", "introduction", "summary", "table of contents"]
            if header.lower() in skip_headers:
                continue

            # Convert header to question
            header_clean = header.strip()
            if "?" in header_clean:
                question = header_clean
            elif header_clean.lower().startswith(("how", "what", "why", "when")):
                question = header_clean + "?"
            else:
                question = f"What is {header_clean} in Temporal?"

            # Truncate long content
            if len(content) > 1500:
                content = content[:1500] + "..."

            qa_pairs.append(
                QAPair(
                    instruction=question,
                    response=content,
                    source_id=doc.id,
                    source_url=doc.url,
                    category="concept",
                    concepts=self.extract_concepts(content),
                )
            )

        return qa_pairs[:5]  # Limit per document

    async def generate_with_llm(self, doc: CollectedDocument) -> list[QAPair]:
        """Generate Q&A pairs using an LLM."""
        if not self.client:
            return self.generate_rule_based(doc)

        prompt = f"""You are creating training data for a language model that will answer questions about Temporal.io (a workflow orchestration platform).

Given the following document, generate 3-5 high-quality question-answer pairs.

Document Title: {doc.title}
Document Type: {doc.content_type}
Source: {doc.source}

Content:
{doc.content[:4000]}

Generate questions that:
1. Cover key concepts explained in the document
2. Include "how-to" questions if code examples are present
3. Ask about best practices or common patterns
4. Are specific to Temporal.io concepts

Return your response as a JSON array with objects containing:
- "instruction": The question
- "response": A comprehensive answer based on the document
- "category": One of "concept", "howto", "code", "troubleshooting"
- "concepts": Array of Temporal concepts covered

JSON Output:"""

        try:
            if self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                )
                result_text = response.choices[0].message.content

            # Parse JSON response
            json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if json_match:
                qa_data = json.loads(json_match.group())
                return [
                    QAPair(
                        instruction=item["instruction"],
                        response=item["response"],
                        source_id=doc.id,
                        source_url=doc.url,
                        category=item.get("category", "concept"),
                        concepts=item.get("concepts", []),
                    )
                    for item in qa_data
                ]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

        # Fallback to rule-based
        return self.generate_rule_based(doc)

    def generate(self, doc: CollectedDocument) -> list[QAPair]:
        """Generate Q&A pairs from a document."""
        if self.use_llm:
            import asyncio
            return asyncio.run(self.generate_with_llm(doc))
        return self.generate_rule_based(doc)

    def process_documents(
        self, documents: list[CollectedDocument], output_path: Path
    ) -> list[QAPair]:
        """Process all documents and generate Q&A pairs."""
        all_qa_pairs = []
        seen_questions = set()

        for doc in documents:
            qa_pairs = self.generate(doc)

            for qa in qa_pairs:
                # Deduplicate by question
                q_normalized = qa.instruction.lower().strip()
                if q_normalized not in seen_questions:
                    seen_questions.add(q_normalized)
                    all_qa_pairs.append(qa)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for qa in all_qa_pairs:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs, saved to {output_path}")
        return all_qa_pairs


def main():
    """Test the Q&A generator."""
    from config import Config

    config = Config()
    generator = QAGenerator(use_llm=False)

    # Load sample documents
    docs_path = config.raw_dir / "temporal_docs.jsonl"
    if docs_path.exists():
        from collectors.base import CollectedDocument

        documents = []
        with open(docs_path, "r") as f:
            for line in f:
                if line.strip():
                    documents.append(CollectedDocument.from_dict(json.loads(line)))

        qa_pairs = generator.process_documents(
            documents[:10], config.processed_dir / "qa_pairs.jsonl"
        )
        print(f"Generated {len(qa_pairs)} Q&A pairs")

        # Show samples
        for qa in qa_pairs[:3]:
            print(f"\nQ: {qa.instruction}")
            print(f"A: {qa.response[:200]}...")


if __name__ == "__main__":
    main()
