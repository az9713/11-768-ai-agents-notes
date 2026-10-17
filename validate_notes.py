"""Focused provenance and package checks for the lecture note set."""

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "sources/manifest.json").read_text(encoding="utf-8"))
    assert [r["lecture"] for r in manifest] == list(range(1, len(manifest) + 1))
    index = BeautifulSoup((ROOT / "index.html").read_text(encoding="utf-8"), "html.parser")
    assert len(index.select(".card")) == len(manifest)
    assert index.find("a", href="development-journey.html") is not None
    assert index.find("a", href="textbook-style-report.html") is not None
    headings = [line for f in sorted(ROOT.glob("lecture-*.md"))
                for line in f.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
    linked = sum("youtube.com/watch" in line for line in headings)
    assert f"{len(headings)} main sections" in index.get_text(" ")
    assert f"{linked} link" in index.get_text(" ")
    journey = BeautifulSoup((ROOT / "development-journey.html").read_text(encoding="utf-8"), "html.parser")
    assert journey.h1 is not None and len(journey.find_all("h2")) >= 8
    for record in manifest:
        number = record["lecture"]
        video_id = record["video_id"]
        prefix = f"lecture-{number:02d}"
        matches = list(ROOT.glob(f"{prefix}-*.md"))
        assert len(matches) == 1, (prefix, matches)
        md_file = matches[0]
        html_file = md_file.with_suffix(".html")
        assert html_file.exists()
        md = md_file.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
        assert soup.h1 is not None and len(soup.find_all("h2")) >= 6
        pagers = soup.select("nav.chapter-nav")
        assert len(pagers) == 2
        expected = {"index.html"}
        if number > 1:
            expected.add(next(ROOT.glob(f"lecture-{number - 1:02d}-*.html")).name)
        if number < len(manifest):
            expected.add(next(ROOT.glob(f"lecture-{number + 1:02d}-*.html")).name)
        assert all({a["href"] for a in nav.find_all("a", href=True)} == expected
                   for nav in pagers), (html_file, expected)
        assert "### Exercises" in md and "### Further reading" in md
        timestamp_lines = [line for line in md.splitlines() if "youtube.com/watch" in line]
        assert len(timestamp_lines) >= 5
        assert all(line.startswith("## ") for line in timestamp_lines)
        assert all(f"watch?v={video_id}" in line for line in timestamp_lines)
        assert "Watch by topic" not in md and "**Extension" not in md
        if number >= 5:
            assert len(md.split()) >= 4500
            assert len(re.findall(r"^```python", md, re.MULTILINE)) >= 2
            research_links = {
                link for link in re.findall(r"\]\((https?://[^)]+)\)", md)
                if "youtube.com/watch" not in link
            }
            assert len(research_links) >= 14
            solutions = md.split("### Solutions and discussion", 1)[1].split(
                "### Further reading", 1
            )[0]
            assert len(solutions.split()) >= 250
            assert len(timestamp_lines) == len(re.findall(r"^## ", md, re.MULTILINE))
        assert record["caption_generated"] is True
        assert record["last_caption_end_seconds"] > 0.98 * record["duration_seconds"]
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("#"):
                assert soup.find(id=href[1:]) is not None, (html_file, href)
            elif not urlparse(href).scheme:
                parsed = urlparse(href)
                target = (ROOT / parsed.path).resolve()
                assert target.exists(), (html_file, href)
                if parsed.fragment and target.suffix == ".html":
                    linked = BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
                    assert linked.find(id=parsed.fragment) is not None, (html_file, href)
            elif "youtube.com/watch" in href:
                assert link.find_parent("h2") is not None, (html_file, href)
                params = parse_qs(urlparse(href).query)
                if params.get("v") == [video_id] and "t" in params:
                    t = int(params["t"][0].removesuffix("s"))
                    assert 0 <= t <= record["last_caption_end_seconds"]
        for heading in soup.find_all("h2"):
            for timestamp in heading.find_all("a", href=True):
                if "youtube.com/watch" in timestamp["href"]:
                    assert re.fullmatch(r"\d{2}:\d{2}", timestamp.get_text(strip=True))
        text = soup.get_text(" ")
        assert "MATHPLACEHOLDER" not in text
        assert len(re.findall(r"\$\$", md)) % 2 == 0
        print(f"{prefix}: {len(md.split())} words, {len(soup.find_all('h2'))} sections, captions {record['segments']} segments")
    required_definitions = {
        5: ("graphics processing unit (GPU)",
            "application programming interface (API)",
            "Stanford Research Institute Problem Solver (STRIPS)"),
        6: ("Fill-in-the-middle (FIM)",
            "application programming interface (API)",
            "document object model (DOM)"),
        7: ("computer-use agent (CUA)",
            "graphical user interface (GUI)",
            "Hypertext markup language (HTML)"),
        8: ("Supervised fine-tuning (SFT)",
            "Reinforcement learning (RL)",
            "Dataset Aggregation"),
        9: ("Reinforcement learning (RL)",
            "Supervised fine-tuning (SFT)",
            "Group Relative Policy Optimization (GRPO)",
            "Proximal Policy Optimization"),
        10: ("Retrieval-augmented generation (RAG)",
             "Reinforcement learning with verifiable rewards (RLVR)",
             "Group Relative Policy Optimization (GRPO)",
             "Dense Passage Retrieval (DPR)"),
    }
    for number, definitions in required_definitions.items():
        md = next(ROOT.glob(f"lecture-{number:02d}-*.md")).read_text(
            encoding="utf-8"
        )
        assert all(definition in md for definition in definitions), (
            number,
            [definition for definition in definitions if definition not in md],
        )
    print("package: PASS")


if __name__ == "__main__":
    main()
