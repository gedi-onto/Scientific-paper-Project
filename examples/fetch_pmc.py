#!/usr/bin/env python
"""Fetch an open-access paper from Europe PMC as pipeline-ready plain text.

    python examples/fetch_pmc.py PMC11578954              # -> PMC11578954.txt
    python examples/fetch_pmc.py PMC11578954 paper.txt

Europe PMC serves full-text JATS XML for open-access articles with no auth and
no bot blocking (unlike publisher sites such as ScienceDirect, which 403). This
pulls the abstract and body prose, drops figures/tables/refs/math, and writes
blank-line-separated paragraphs -- exactly what ``split_paragraphs`` expects.
"""

import re
import sys
import urllib.request
from xml.etree import ElementTree

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Structural elements whose text is not running prose.
SKIP = {"xref", "table-wrap", "fig", "disp-formula", "inline-formula", "graphic", "media"}


def paragraph_text(node) -> str:
    """Flatten one <p> to plain text, dropping citation markers and float content."""
    parts = []
    if node.text:
        parts.append(node.text)
    for child in node:
        tag = child.tag.split("}")[-1]
        if tag not in SKIP:
            parts.append(paragraph_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def extract(xml: bytes) -> str:
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for section in ("abstract", "body"):
        for parent in root.iter():
            if parent.tag.split("}")[-1] != section:
                continue
            for p in parent.iter():
                if p.tag.split("}")[-1] == "p":
                    text = re.sub(r"\s+", " ", paragraph_text(p)).strip()
                    if len(text) > 80:  # skip captions / one-line fragments
                        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def main() -> None:
    pmcid = sys.argv[1] if len(sys.argv) > 1 else "PMC11578954"
    out = sys.argv[2] if len(sys.argv) > 2 else f"{pmcid}.txt"
    with urllib.request.urlopen(EUROPE_PMC.format(pmcid=pmcid)) as response:
        text = extract(response.read())
    if not text:
        raise SystemExit(f"{pmcid}: no open-access full text (abstract-only or closed).")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"{pmcid}: {len(text.split())} words, {text.count(chr(10) * 2) + 1} paragraphs -> {out}")


if __name__ == "__main__":
    main()
