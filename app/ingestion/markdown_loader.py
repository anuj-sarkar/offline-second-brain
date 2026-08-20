"""
app/ingestion/markdown_loader.py

Loader for Markdown (.md) and Plain Text (.txt) files.
Enables Offline Second Brain to ingest Obsidian vaults, notes, and research summaries.
Detects Markdown headers (#, ##, ###) to preserve structural section metadata.
"""

import logging
import re

logger = logging.getLogger(__name__)


def parse_markdown_sections(text: str) -> list[dict]:
    """
    Split markdown into logical sections based on # / ## / ### headers.
    Returns list of dicts with 'section_title' and 'text'.
    """
    lines = text.split("\n")
    sections = []
    current_title = "Document"
    current_lines = []

    header_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({
                        "section_title": current_title,
                        "text": body,
                    })
                current_lines = []
            current_title = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "section_title": current_title,
                "text": body,
            })

    if not sections and text.strip():
        sections.append({
            "section_title": "Document",
            "text": text.strip(),
        })

    return sections
