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
SITE_BASE_PATH = "/OrbitBreakersBench/"


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []
        self.images: list[dict[str, str | None]] = []
        self.ids: list[str] = []
        self.tab_controls: list[str] = []
        self.graph_link_labels: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if tag in {"a", "link"} and attributes.get("href"):
            self.targets.append(attributes["href"])
        if (
            tag == "a"
            and "figure-canvas" in (attributes.get("class") or "").split()
            and attributes.get("aria-label")
        ):
            self.graph_link_labels.append(attributes["aria-label"])
        if tag in {"script", "img"} and attributes.get("src"):
            self.targets.append(attributes["src"])
        if tag == "source" and attributes.get("srcset"):
            self.targets.append(attributes["srcset"])
        if tag == "img":
            self.images.append(attributes)
        if attributes.get("role") == "tab" and attributes.get("aria-controls"):
            self.tab_controls.append(attributes["aria-controls"])


def resolve_internal(page: Path, artifact: Path, target: str) -> Path | None:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:", "tel:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    if path.startswith("/"):
        if not path.startswith(SITE_BASE_PATH):
            raise AssertionError(f"Unexpected absolute path: {page} -> {target}")
        resolved = (artifact / path.removeprefix(SITE_BASE_PATH)).resolve()
    else:
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
        assert "What we learned from benchmarking agents—and then optimizing the experts" in index_html
        assert "Three result axes. Two hard boundaries." in index_html
        assert "2.88×" in index_html
        assert "10/12 → 11/12†" in index_html
        assert "role=\"tablist\"" in index_html
        assert index_html.count("data-graph-tab") == 5
        assert "Figures from the final report" in index_html
        assert "Accessible data &amp; public evidence" in index_html
        assert "Task-level final validity and artifact runtime" in index_html
        assert "Solver-resource comparison" in index_html
        assert "126.675 s" in index_html and "123.188 s" in index_html
        assert "Task 08&#x27;s independent 64-GiB paired session" in index_html
        assert "https://github.com/sxzgroup/ORBIT-Q/pull/4" in index_html
        assert "https://github.com/sxzgroup/ORBIT-Q/pull/5" in index_html
        assert "https://github.com/sxzgroup/ORBIT-Q/pull/6" in index_html
        assert "45.758× Task 07†" not in index_html
        assert "≈35% lower slowdown" in index_html
        assert "Search tasks" in index_html
        for index in range(1, 13):
            assert f"task-{index:02d}/" in index_html

        task05 = (artifact / "tasks" / "task-05" / "index.html").read_text(encoding="utf-8")
        assert "1.939×‡" in task05
        assert "1.105×–2.772×" in task05
        assert "Why this supersedes the earlier 14.08× row" in task05
        assert "sxzgroup/ORBIT-Q/pull/19" in task05
        assert "optimized_solutions/challenge-05/solution_5.py" in task05
        assert "tasks/challenge-05/solution/solution_5.py" in task05
        assert "b3b0c08f0bbc82470da5cbe39a6ba31e39f6a7fe" in task05
        assert "14.076×" not in task05

        task06 = (artifact / "tasks" / "task-06" / "index.html").read_text(encoding="utf-8")
        assert "1.504×‡" in task06
        assert "1.489×–1.520×" in task06
        assert "requires one additional passing matched pair" in task06
        assert ">Pending<" not in task06

        task08 = (artifact / "tasks" / "task-08" / "index.html").read_text(encoding="utf-8")
        assert "OOM → PASS" in task08
        assert "no numerical speedup claim" in task08.lower()
        assert "126.675 → 123.188 s" in task08
        assert "0.818×–1.273×" in task08

        task10 = (artifact / "tasks" / "task-10" / "index.html").read_text(encoding="utf-8")
        assert "4.898×" in task10
        assert "4.598×–5.199×" in task10
        assert "Supplemental sixth pair" in task10
        assert "4.940×" in task10

        task12 = (artifact / "tasks" / "task-12" / "index.html").read_text(encoding="utf-8")
        assert "3.914×†" in task12
        assert "3.877×–3.951×" in task12
        assert "Tracked secondary variant" in task12
        assert "4.248×" in task12
        assert "src/solutions/task-12/solution_12.py" in task12
        assert "solution_12_fused.py" not in task12
        assert "promoted primary" not in task12.lower()

        expected_figures = {
            "gpt55-vs-gpt56-comparison.png",
            "gpt56sol-high-agent-resource-use.png",
            "gpt56sol-high-vs-ultra-outcomes.png",
            "gpt56sol-high-vs-ultra-resources.png",
            "gpt56sol-ultra-agent-resource-use.png",
            "fable5-runtime-ratios.png",
            "fable5-runtime-ratios.svg",
            "expert-optimization-runtime-log-bars.png",
            "expert-optimization-runtime-log-bars.svg",
            "factor-ablation-overview.png",
            "factor-ablation-overview.svg",
        }
        assert {path.name for path in (artifact / "figures").iterdir()} == expected_figures

        for index in range(1, 13):
            patch = artifact / "diffs" / f"task-{index:02d}.diff"
            assert patch.is_file() and patch.stat().st_size > 100
            text = patch.read_text(encoding="utf-8")
            if index == 5:
                assert text.startswith(
                    "--- a/tasks/challenge-05/solution/solution_5.py"
                )
                assert "-Task Suite Problem 5" not in text
            else:
                assert text.startswith("--- a/references/")
            assert "\n+++ b/" in text
            focused = artifact / "diffs" / f"task-{index:02d}.focused.diff"
            assert focused.is_file() and 100 < focused.stat().st_size < 20000

        for page in sorted(artifact.rglob("*.html")):
            source = page.read_text(encoding="utf-8")
            assert "file://" not in source
            collector = LinkCollector()
            collector.feed(source)
            assert len(collector.ids) == len(set(collector.ids)), (
                f"Duplicate HTML id in {page.relative_to(artifact)}"
            )
            for control in collector.tab_controls:
                assert control in collector.ids, (
                    f"Tab control has no panel in {page.relative_to(artifact)}: {control}"
                )
            for image in collector.images:
                assert image.get("alt"), (
                    f"Missing image alt text in {page.relative_to(artifact)}"
                )
                assert image.get("width") and image.get("height"), (
                    f"Image dimensions missing in {page.relative_to(artifact)}"
                )
            for target in collector.targets:
                resolved = resolve_internal(page, artifact, target)
                if resolved is not None:
                    assert resolved.exists(), f"Broken link: {page.relative_to(artifact)} -> {target}"
            if page == artifact / "index.html":
                assert len(collector.graph_link_labels) == 8
                assert len(set(collector.graph_link_labels)) == 8

        not_found = artifact / "404.html"
        not_found_source = not_found.read_text(encoding="utf-8")
        not_found_collector = LinkCollector()
        not_found_collector.feed(not_found_source)
        nested_unknown = artifact / "missing" / "deep" / "index.html"
        for target in not_found_collector.targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            assert parsed.path.startswith(SITE_BASE_PATH), (
                f"404 target is not project-root absolute: {target}"
            )
            resolved = resolve_internal(nested_unknown, artifact, target)
            assert resolved is not None and resolved.exists(), (
                f"Nested 404 target is broken: {target}"
            )

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Visual report" in readme
        assert "tabbed, expandable graph gallery" in readme
        assert "docs/figures/gpt55-vs-gpt56-comparison.png" in readme
        assert "docs/figures/expert-optimization-runtime-log-bars.png" in readme
        assert "docs/figures/factor-ablation-overview.png" in readme

        tasks_payload = json.loads(
            (ROOT / "site" / "data" / "tasks.json").read_text(encoding="utf-8")
        )
        task05_data = next(
            task for task in tasks_payload["tasks"] if task["id"] == "05"
        )
        assert task05_data["reference_sha256"] == (
            "212ec101f72aa87468b5d2aabc76b445b158a69d8f99e709cb8a0bbd05476a8f"
        )
        assert task05_data["candidate_sha256"] == (
            "bbe5768d9e3b0e1e7e28d611ad2be8392a8bc3e031fb662604adca90a9f08b79"
        )

        factor_svg = (
            artifact / "figures" / "factor-ablation-overview.svg"
        ).read_text(encoding="utf-8")
        assert "Factor evidence for all twelve" in factor_svg
        assert "Euler cold compile" in factor_svg
        assert "BCOO microbench" in factor_svg
        assert "3.109×*" in factor_svg
        assert ">supported<" in factor_svg

        assert (artifact / ".nojekyll").is_file()
        assert (artifact / "og.png").stat().st_size > 100000
        assert (artifact / "sitemap.xml").is_file()
        assert (artifact / "llms.txt").is_file()
        assert (artifact / "assets" / "styles.css").stat().st_size > 10000
        assert (artifact / "assets" / "app.js").stat().st_size > 1000

    print(
        "Static site checks passed: 15 HTML routes, 12 diffs, 11 report figures, "
        "links, tabs, search, and final-report claim boundaries."
    )


if __name__ == "__main__":
    main()
