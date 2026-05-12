"""
parser.py
Parses DITA XML files into structured chunks with metadata.
Each section within a topic becomes one chunk.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict


def extract_text(element) -> str:
    """Recursively extract all text content from an XML element."""
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def parse_dita_file(filepath: str) -> List[Dict]:
    """
    Parse a DITA file and return a list of chunks.
    Each chunk is one section with metadata.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    topic_id = root.get("id", "unknown")
    topic_title = root.findtext("title", default="Untitled")
    topic_shortdesc = root.findtext("shortdesc", default="")

    filename = Path(filepath).stem
    chunks = []

    body = root.find("body")
    if body is None:
        return chunks

    for section in body.findall("section"):
        section_id = section.get("id", "unknown")
        section_title = section.findtext("title", default="Untitled Section")
        section_text = extract_text(section)

        # Remove the title text from the body text to avoid duplication
        section_text = section_text.replace(section_title, "", 1).strip()

        chunks.append({
            "chunk_id": f"{filename}_{section_id}",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "topic_shortdesc": topic_shortdesc,
            "section_id": section_id,
            "section_title": section_title,
            "text": section_text,
            "source_file": filename,
            "tags": generate_tags(topic_title, section_title),
        })

    return chunks


def generate_tags(topic_title: str, section_title: str) -> str:
    """Generate simple keyword tags from titles for metadata enrichment."""
    combined = f"{topic_title} {section_title}".lower()
    keywords = []
    tag_map = {
        "install": "installation",
        "config": "configuration",
        "troubleshoot": "troubleshooting",
        "license": "licensing",
        "performance": "performance",
        "workspace": "workspace",
        "unit": "units",
        "crash": "crashes",
        "error": "errors",
        "download": "download",
        "system": "system-requirements",
        "user": "user-preferences",
    }
    for keyword, tag in tag_map.items():
        if keyword in combined:
            keywords.append(tag)
    return ", ".join(keywords) if keywords else "general"


def parse_all_dita_files(data_dir: str) -> List[Dict]:
    """Parse all .dita files in a directory."""
    data_path = Path(data_dir)
    all_chunks = []
    for dita_file in sorted(data_path.glob("*.dita")):
        chunks = parse_dita_file(str(dita_file))
        all_chunks.extend(chunks)
        print(f"Parsed {dita_file.name}: {len(chunks)} chunks")
    return all_chunks
