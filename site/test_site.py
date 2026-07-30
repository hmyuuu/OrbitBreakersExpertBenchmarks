#!/usr/bin/env python3
"""Static artifact checks for the OrbitBreakers results site."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"a", "link"} and attributes.get("href"):
            self.targets.append(attributes["href"])
        if tag in {"script", "img"} and attributes.get("src"):
            self.targets.append(attributes["src"])


def resolve_internal(page: Path, artifact: Path, target: str) -> Path | None:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:", "tel:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    resolved = (page.parent / path).resolve()
    if path.endswith("/") or resolved.is_dir():
        resolved = resolved / "index.html"
    if artifact.resolve() not in resolved.parents and resolved != artifact.resolve():
        raise AssertionError(f"Link escapes artifact: {page} -> {target}")
    return resolved


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "site" / "build.py")],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["node", "--check", str(ROOT / "site" / "static" / "app.js")],
        check=True,
        cwd=ROOT,
    )

    with tempfile.TemporaryDirectory(prefix="orbitbreakers-site-test-") as temp:
        artifact = Path(temp) / "artifact"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "site" / "build.py"),
                "--output",
                str(artifact),
            ],
            check=True,
            cwd=ROOT,
        )

        expected_pages = {
            artifact / "index.html",
            artifact / "404.html",
            artifact / "methodology" / "index.html",
            *{
                artifact / "tasks" / f"task-{index:02d}" / "index.html"
                for index in range(1, 13)
            },
        }
        missing = sorted(str(path) for path in expected_pages if not path.is_file())
        assert not missing, f"Missing expected pages: {missing}"

        search = json.loads((artifact / "search-index.json").read_text(encoding="utf-8"))
        assert len(search) == 12
        assert [entry["id"] for entry in search] == [f"{index:02d}" for index in range(1, 13)]

        index_html = (artifact / "index.html").read_text(encoding="utf-8")
        assert "Break the runtime." in index_html
        assert "Every task, one evidence row." in index_html
        for index in range(1, 13):
            assert f"task-{index:02d}/" in index_html

        task06 = (artifact / "tasks" / "task-06" / "index.html").read_text(encoding="utf-8")
        assert "One pair short" in task06
        assert "not as validated SOTA" in task06

        task08 = (artifact / "tasks" / "task-08" / "index.html").read_text(encoding="utf-8")
        assert "OOM → PASS" in task08
        assert "no numerical speedup claim" in task08.lower()

        task12 = (artifact / "tasks" / "task-12" / "index.html").read_text(encoding="utf-8")
        assert "solution_12_fused.py" in task12
        assert "tracked fused variant" in task12.lower()

        for index in range(1, 13):
            patch = artifact / "diffs" / f"task-{index:02d}.diff"
            assert patch.is_file() and patch.stat().st_size > 100
            text = patch.read_text(encoding="utf-8")
            assert text.startswith("--- a/references/")
            assert "\n+++ b/" in text
            focused = artifact / "diffs" / f"task-{index:02d}.focused.diff"
            assert focused.is_file() and 100 < focused.stat().st_size < 20000

        for page in sorted(artifact.rglob("*.html")):
            source = page.read_text(encoding="utf-8")
            assert "file://" not in source
            assert 'href="/' not in source
            assert 'src="/' not in source
            collector = LinkCollector()
            collector.feed(source)
            for target in collector.targets:
                resolved = resolve_internal(page, artifact, target)
                if resolved is not None:
                    assert resolved.exists(), f"Broken link: {page.relative_to(artifact)} -> {target}"

        assert (artifact / ".nojekyll").is_file()
        assert (artifact / "og.png").stat().st_size > 100000
        assert (artifact / "sitemap.xml").is_file()
        assert (artifact / "llms.txt").is_file()
        assert (artifact / "assets" / "styles.css").stat().st_size > 10000
        assert (artifact / "assets" / "app.js").stat().st_size > 1000

    print("Static site checks passed: 15 HTML routes, 12 diffs, links, search, and claim boundaries.")


if __name__ == "__main__":
    main()
