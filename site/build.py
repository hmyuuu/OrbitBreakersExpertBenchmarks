#!/usr/bin/env python3
"""Build the dependency-free OrbitBreakers GitHub Pages artifact."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
DATA_PATH = SITE_DIR / "data" / "tasks.json"
REPORT_PATH = SITE_DIR / "data" / "report.json"
STATIC_DIR = SITE_DIR / "static"
FIGURES_DIR = ROOT / "docs" / "figures"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_data() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    try:
        payload["site"]["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    tasks = payload["tasks"]
    if len(tasks) != 12:
        raise ValueError(f"Expected exactly 12 tasks, found {len(tasks)}")
    ids = [task["id"] for task in tasks]
    if ids != [f"{index:02d}" for index in range(1, 13)]:
        raise ValueError(f"Task IDs must be 01 through 12 in order, got {ids}")
    if len(report.get("evidence_axes", [])) != 3:
        raise ValueError("Final report must define exactly three evidence axes")
    if len(report.get("graph_tabs", [])) < 3:
        raise ValueError("Final report must define at least three graph tabs")
    for tab in report["graph_tabs"]:
        if not tab.get("figures"):
            raise ValueError(f'Graph tab {tab.get("id", "<missing>")} has no figures')
        for figure in tab["figures"]:
            for key in ("src", "width", "height", "alt", "caption"):
                if not figure.get(key):
                    raise ValueError(
                        f'Graph figure in {tab["id"]} is missing required field {key}'
                    )
            if not (FIGURES_DIR / figure["src"]).is_file():
                raise FileNotFoundError(FIGURES_DIR / figure["src"])
            if figure.get("svg") and not (FIGURES_DIR / figure["svg"]).is_file():
                raise FileNotFoundError(FIGURES_DIR / figure["svg"])
    for task in tasks:
        candidate_source = ROOT / task.get(
            "candidate_source_path", task["candidate_path"]
        )
        if not candidate_source.is_file():
            raise FileNotFoundError(candidate_source)
        expected_sha256 = task.get("candidate_sha256")
        if expected_sha256:
            actual_sha256 = hashlib.sha256(candidate_source.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Candidate snapshot hash mismatch for Task {task['id']}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        reference_source = ROOT / task.get(
            "reference_source_path", task["reference_path"]
        )
        if not reference_source.is_file():
            raise FileNotFoundError(reference_source)
        expected_reference_sha256 = task.get("reference_sha256")
        if expected_reference_sha256:
            actual_reference_sha256 = hashlib.sha256(
                reference_source.read_bytes()
            ).hexdigest()
            if actual_reference_sha256 != expected_reference_sha256:
                raise ValueError(
                    f"Reference snapshot hash mismatch for Task {task['id']}: "
                    f"expected {expected_reference_sha256}, "
                    f"got {actual_reference_sha256}"
                )
    return payload["site"], report, tasks


def status_badge(task: dict[str, Any]) -> str:
    status = task["status"]
    return (
        f'<span class="status-badge status-{escape(status["kind"])}">'
        f'<span class="status-dot" aria-hidden="true"></span>'
        f'{escape(status["label"])}</span>'
    )


def task_href(root_prefix: str, task: dict[str, Any]) -> str:
    return f'{root_prefix}tasks/{escape(task["slug"])}/'


def repo_file_url(site: dict[str, Any], path: str) -> str:
    return f'{site["repository"]}/blob/main/{quote(path)}'


def site_base_path(site: dict[str, Any]) -> str:
    path = urlsplit(site["site_url"]).path or "/"
    return path if path.endswith("/") else f"{path}/"


def common_head(
    *,
    site: dict[str, Any],
    title: str,
    description: str,
    canonical_path: str,
    root_prefix: str,
) -> str:
    canonical = f'{site["site_url"]}{canonical_path}'
    og_path = f"{site['site_url']}og.png"
    return f"""<!doctype html>
<html lang="en" data-theme="system">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <meta name="theme-color" content="#fbfaf6" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#0f1220" media="(prefers-color-scheme: dark)">
  <title>{escape(title)}</title>
  <meta name="description" content="{escape(description)}">
  <link rel="canonical" href="{escape(canonical)}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{escape(title)}">
  <meta property="og:description" content="{escape(description)}">
  <meta property="og:url" content="{escape(canonical)}">
  <meta property="og:image" content="{escape(og_path)}">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="stylesheet" href="{root_prefix}assets/styles.css">
  <script>
    (() => {{
      const saved = localStorage.getItem("ob-theme");
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.dataset.theme = saved || (dark ? "dark" : "light");
    }})();
  </script>
</head>
"""


def header(site: dict[str, Any], root_prefix: str, *, docs: bool = False) -> str:
    mobile_button = (
        '<button class="icon-button sidebar-toggle" type="button" '
        'aria-label="Open task navigation" aria-expanded="false">'
        '<span aria-hidden="true">☰</span></button>'
        if docs
        else ""
    )
    return f"""
<a class="skip-link" href="#main-content">Skip to content</a>
<header class="site-header">
  <div class="header-inner">
    <div class="header-leading">
      {mobile_button}
      <a class="brand" href="{root_prefix}">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <span class="brand-copy"><strong>OrbitBreakers</strong><small>Evidence report</small></span>
      </a>
    </div>
    <nav class="header-nav" aria-label="Primary">
      <a href="{root_prefix}" data-nav="results">Summary</a>
      <a href="{root_prefix}#figures">Figures</a>
      <a href="{root_prefix}#task-index">Tasks</a>
      <a href="{root_prefix}methodology/" data-nav="methodology">Protocol</a>
    </nav>
    <div class="header-actions">
      <button class="search-trigger" type="button" data-open-search>
        <span aria-hidden="true">⌕</span><span>Search tasks</span><kbd>⌘ K</kbd>
      </button>
      <button class="icon-button theme-toggle" type="button" aria-label="Toggle color theme">
        <span class="theme-sun" aria-hidden="true">☼</span>
        <span class="theme-moon" aria-hidden="true">◐</span>
      </button>
      <a class="icon-button github-link" href="{escape(site["repository"])}" aria-label="Open GitHub repository">
        <span aria-hidden="true">GH</span>
      </a>
    </div>
  </div>
</header>
"""


def search_dialog(tasks: list[dict[str, Any]], root_prefix: str) -> str:
    search_payload = [
        {
            "id": task["id"],
            "title": task["title"],
            "summary": task["summary"],
            "metric": task["metric"]["headline"],
            "status": task["status"]["label"],
            "href": task_href(root_prefix, task),
        }
        for task in tasks
    ]
    encoded = json.dumps(search_payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""
<dialog class="search-dialog" id="search-dialog" aria-labelledby="search-title">
  <div class="search-panel">
    <div class="search-box">
      <span aria-hidden="true">⌕</span>
      <label class="sr-only" for="task-search">Search tasks</label>
      <input id="task-search" type="search" role="combobox" aria-controls="search-results" aria-expanded="false" aria-autocomplete="list" placeholder="Search a task, method, or result…" autocomplete="off">
      <button type="button" class="search-close" data-close-search aria-label="Close search">Esc</button>
    </div>
    <p class="search-label" id="search-title">Task search</p>
    <div class="search-results" id="search-results" role="listbox"></div>
    <p class="search-empty" id="search-empty" hidden>No task matches that search.</p>
    <div class="search-help"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span></div>
  </div>
</dialog>
<script type="application/json" id="task-search-data">{encoded}</script>
"""


def footer(site: dict[str, Any], root_prefix: str) -> str:
    return f"""
<footer class="site-footer">
  <div>
    <strong>OrbitBreakers Bench</strong>
    <p>Claims stay scoped to the public workload and recorded measurement environment.</p>
  </div>
  <div class="footer-links">
    <a href="{root_prefix}methodology/">Measurement protocol</a>
    <a href="{escape(site["repository"])}">Source on GitHub</a>
  </div>
  <p class="footer-meta">Evidence index · {escape(site["updated"])} · {escape(site["commit"])}</p>
</footer>
"""


def sidebar(
    site: dict[str, Any],
    tasks: list[dict[str, Any]],
    root_prefix: str,
    *,
    current: str | None = None,
    methodology_current: bool = False,
) -> str:
    task_links = []
    for task in tasks:
        current_attr = ' aria-current="page"' if current == task["slug"] else ""
        active = " is-active" if current == task["slug"] else ""
        task_links.append(
            f'<a class="sidebar-task{active}" href="{task_href(root_prefix, task)}"{current_attr}>'
            f'<span class="sidebar-id">{escape(task["id"])}</span>'
            f'<span class="sidebar-title">{escape(task["short_title"])}</span>'
            f'<span class="sidebar-status status-{escape(task["status"]["kind"])}" '
            f'title="{escape(task["status"]["label"])}"></span></a>'
        )
    methodology_active = " is-active" if methodology_current else ""
    return f"""
<aside class="docs-sidebar" id="docs-sidebar">
  <nav aria-label="Documentation">
    <p class="sidebar-label">Overview</p>
    <a class="sidebar-overview" href="{root_prefix}">Result index</a>
    <a class="sidebar-overview{methodology_active}" href="{root_prefix}methodology/"{
        ' aria-current="page"' if methodology_current else ''
    }>Measurement protocol</a>
    <p class="sidebar-label sidebar-label-tasks">Tasks</p>
    <div class="sidebar-tasks">
      {''.join(task_links)}
    </div>
  </nav>
  <div class="sidebar-footer">
    <span>12 public workloads</span>
    <a href="{escape(site["repository"])}">View repository ↗</a>
  </div>
</aside>
<div class="sidebar-scrim" data-close-sidebar></div>
"""


def diff_patch(task: dict[str, Any]) -> str:
    reference = ROOT / task.get("reference_source_path", task["reference_path"])
    candidate = ROOT / task.get("candidate_source_path", task["candidate_path"])
    before = reference.read_text(encoding="utf-8").splitlines(keepends=True)
    after = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f'a/{task.get("reference_diff_path", task["reference_path"])}',
            tofile=f'b/{task["candidate_path"]}',
            n=3,
        )
    )


def focused_diff(task: dict[str, Any], patch: str) -> str:
    lines = patch.splitlines(keepends=True)
    if not lines:
        return "No source difference recorded.\n"
    headers: list[str] = []
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            current = [line]
            hunks.append(current)
        elif current is None:
            headers.append(line)
        else:
            current.append(line)

    keywords = [keyword.lower() for keyword in task.get("diff_keywords", [])]

    def score(hunk: list[str]) -> tuple[int, int]:
        text = "".join(hunk).lower()
        keyword_score = sum(100 for keyword in keywords if keyword in text)
        changed = sum(1 for line in hunk if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
        return keyword_score + changed, changed

    best = max(hunks, key=score) if hunks else lines
    if len(best) <= 92:
        return "".join(headers[:2] + best)

    center = None
    for keyword in keywords:
        matching = [
            index for index, line in enumerate(best) if keyword in line.lower()
        ]
        if matching:
            center = matching[0]
            break
    center = center if center is not None else len(best) // 2
    start = max(1, center - 26)
    end = min(len(best), center + 39)
    marker = "@@ focused excerpt · surrounding patch lines omitted @@\n"
    return "".join(headers[:2] + [marker] + best[start:end])


def diff_markup(patch: str) -> str:
    rendered = []
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            css_class = "diff-file"
        elif line.startswith("@@"):
            css_class = "diff-hunk"
        elif line.startswith("+"):
            css_class = "diff-add"
        elif line.startswith("-"):
            css_class = "diff-remove"
        else:
            css_class = "diff-context"
        rendered.append(f'<span class="{css_class}">{escape(line)}</span>')
    return "\n".join(rendered)


def metric_bar(task: dict[str, Any]) -> str:
    metric = task["metric"]
    baseline_value = metric["baseline_value"]
    candidate_value = metric["candidate_value"]
    if baseline_value is None:
        return f"""
<div class="runtime-bars runtime-bars-oom" aria-label="Reference exhausted memory; candidate passed">
  <div class="runtime-row"><span>Original</span><div class="runtime-track"><i class="bar-oom"></i></div><strong>{escape(metric["baseline"])}</strong></div>
  <div class="runtime-row"><span>Optimized</span><div class="runtime-track"><i class="bar-candidate" style="width: 42%"></i></div><strong>{escape(metric["candidate"])}</strong></div>
</div>
"""
    percentage = max(2.0, min(100.0, 100.0 * candidate_value / baseline_value))
    return f"""
<div class="runtime-bars" aria-label="Mean runtime comparison">
  <div class="runtime-row"><span>Original</span><div class="runtime-track"><i class="bar-reference" style="width: 100%"></i></div><strong>{escape(metric["baseline"])}</strong></div>
  <div class="runtime-row"><span>Optimized</span><div class="runtime-track"><i class="bar-candidate" style="width: {percentage:.2f}%"></i></div><strong>{escape(metric["candidate"])}</strong></div>
</div>
"""


def measurement_context(task: dict[str, Any]) -> str:
    context = task.get("measurement_context")
    if not context:
        return ""
    items = "".join(
        f"<div><dt>{escape(item['label'])}</dt><dd>{escape(item['value'])}</dd></div>"
        for item in context["items"]
    )
    return f"""
<aside class="measurement-context" aria-labelledby="measurement-context-{escape(task["id"])}">
  <div>
    <p class="eyebrow">Measurement context</p>
    <h2 id="measurement-context-{escape(task["id"])}">{escape(context["title"])}</h2>
    <p>{escape(context["note"])}</p>
  </div>
  <dl>{items}</dl>
</aside>
"""


def outcome_cards(report: dict[str, Any]) -> str:
    cards = []
    for outcome in report["outcomes"]:
        cards.append(
            f"""
<article class="outcome-card outcome-{escape(outcome["tone"])}">
  <p class="eyebrow">{escape(outcome["eyebrow"])}</p>
  <strong>{escape(outcome["value"])}</strong>
  <span>{escape(outcome["label"])}</span>
  <p>{escape(outcome["detail"])}</p>
</article>
"""
        )
    boundaries = "".join(
        f"""
<article>
  <span>{escape(item["label"])}</span>
  <strong>{escape(item["value"])}</strong>
  <small>{escape(item["detail"])}</small>
</article>
"""
        for item in report["boundary_results"]
    )
    return f"""
<section class="report-outcomes section-shell" aria-labelledby="outcomes-heading">
  <div class="section-heading report-section-heading">
    <div>
      <p class="eyebrow">Final synthesis</p>
      <h2 id="outcomes-heading">Three result axes. Two hard boundaries.</h2>
    </div>
    <p>Agent validity, artifact runtime, and expert optimization answer different questions. The final report does not collapse them into one leaderboard.</p>
  </div>
  <div class="outcome-grid">{''.join(cards)}</div>
  <div class="boundary-rail" aria-label="Results requiring special interpretation">{boundaries}</div>
</section>
"""


def evidence_model(report: dict[str, Any]) -> str:
    axes = "".join(
        f"""
<article>
  <span>{escape(axis["number"])}</span>
  <h3>{escape(axis["title"])}</h3>
  <p>{escape(axis["question"])}</p>
  <code>{escape(axis["metric"])}</code>
  <small>{escape(axis["direction"])}</small>
</article>
"""
        for axis in report["evidence_axes"]
    )
    return f"""
<section class="evidence-model section-shell" id="evidence-model" aria-labelledby="evidence-model-heading">
  <div class="section-heading report-section-heading">
    <div>
      <p class="eyebrow">Evidence model</p>
      <h2 id="evidence-model-heading">Read the denominator before the ratio.</h2>
    </div>
    <p>Absolute runtimes from different tasks or hosts are never pooled. Every ratio stays attached to its same-task comparison and stated environment.</p>
  </div>
  <div class="evidence-axis-grid">{axes}</div>
</section>
"""


def graph_figure(figure: dict[str, Any], *, eager: bool) -> str:
    source = (
        f'<source srcset="figures/{escape(figure["svg"])}" type="image/svg+xml">'
        if figure.get("svg")
        else ""
    )
    full_size = figure.get("svg", figure["src"])
    tall = " is-tall" if figure["height"] > figure["width"] else ""
    loading = "eager" if eager else "lazy"
    priority = ' fetchpriority="high"' if eager else ""
    return f"""
<figure class="report-figure{tall}">
  <a class="figure-canvas" href="figures/{escape(full_size)}" aria-label="Open full-size graph: {escape(figure["alt"])}">
    <picture>
      {source}
      <img src="figures/{escape(figure["src"])}" width="{escape(figure["width"])}" height="{escape(figure["height"])}" loading="{loading}" decoding="async"{priority} alt="{escape(figure["alt"])}">
    </picture>
  </a>
  <figcaption>
    <span>{escape(figure["caption"])}</span>
    <a href="figures/{escape(full_size)}">Open full size ↗</a>
  </figcaption>
</figure>
"""


def graph_data_details(tab: dict[str, Any]) -> str:
    tables = []
    for table in tab.get("data_tables", []):
        headers = "".join(
            f'<th scope="col">{escape(header)}</th>' for header in table["headers"]
        )
        rows = "".join(
            "<tr>"
            + "".join(
                f'<th scope="row">{escape(value)}</th>'
                if column_index == 0
                else f"<td>{escape(value)}</td>"
                for column_index, value in enumerate(row)
            )
            + "</tr>"
            for row in table["rows"]
        )
        tables.append(
            f"""
<section class="graph-data-table">
  <h4>{escape(table["title"])}</h4>
  <div class="graph-data-scroll">
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p>{escape(table["note"])}</p>
</section>
"""
        )
    sources = "".join(
        f'<a href="{escape(source["url"])}">{escape(source["label"])} <span aria-hidden="true">↗</span></a>'
        for source in tab.get("evidence", [])
    )
    if not tables and not sources:
        return ""
    return f"""
<details class="graph-data">
  <summary><span>Accessible data &amp; public evidence</span><small>{len(tables)} table{"s" if len(tables) != 1 else ""} · {len(tab.get("evidence", []))} source{"s" if len(tab.get("evidence", [])) != 1 else ""}</small></summary>
  <div class="graph-data-body">
    {''.join(tables)}
    <nav class="graph-sources" aria-label="{escape(tab["label"])} public evidence">{sources}</nav>
  </div>
</details>
"""


def graph_gallery(report: dict[str, Any]) -> str:
    buttons = []
    panels = []
    for index, tab in enumerate(report["graph_tabs"]):
        active = index == 0
        tab_id = f'graph-tab-{tab["id"]}'
        panel_id = f'graph-panel-{tab["id"]}'
        buttons.append(
            f'<button type="button" role="tab" id="{escape(tab_id)}" '
            f'aria-controls="{escape(panel_id)}" aria-selected="{str(active).lower()}" '
            f'tabindex="{0 if active else -1}" data-graph-tab>{escape(tab["label"])}</button>'
        )
        stats = "".join(f"<li>{escape(stat)}</li>" for stat in tab["stats"])
        figures = "".join(
            graph_figure(figure, eager=active and figure_index == 0)
            for figure_index, figure in enumerate(tab["figures"])
        )
        hidden = "" if active else " hidden"
        panels.append(
            f"""
<section class="graph-panel" role="tabpanel" id="{escape(panel_id)}" aria-labelledby="{escape(tab_id)}"{hidden}>
  <div class="graph-panel-copy">
    <div>
      <p class="eyebrow">{escape(tab["eyebrow"])}</p>
      <h3>{escape(tab["title"])}</h3>
      <p>{escape(tab["summary"])}</p>
    </div>
    <ul aria-label="Key graph facts">{stats}</ul>
  </div>
  <div class="graph-figure-grid graph-figure-count-{len(tab["figures"])}">{figures}</div>
  {graph_data_details(tab)}
</section>
"""
        )
    return f"""
<section class="graph-gallery section-shell" id="figures" aria-labelledby="figures-heading">
  <div class="section-heading report-section-heading">
    <div>
      <p class="eyebrow">Figures from the final report</p>
      <h2 id="figures-heading">Inspect the evidence, one view at a time.</h2>
    </div>
    <p>These are the report's generated graphs—not decorative mockups. Open any figure at full resolution to inspect task labels and caveats.</p>
  </div>
  <div class="graph-shell">
    <div class="graph-tablist" role="tablist" aria-label="Final-report graph groups">{''.join(buttons)}</div>
    {''.join(panels)}
  </div>
  <noscript><style>.graph-panel[hidden]{{display:block !important}}</style></noscript>
</section>
"""


def report_findings(report: dict[str, Any]) -> str:
    findings = "".join(
        f"""
<article>
  <span>{escape(finding["number"])}</span>
  <h3>{escape(finding["title"])}</h3>
  <p>{escape(finding["body"])}</p>
</article>
"""
        for finding in report["findings"]
    )
    return f"""
<section class="report-findings section-shell" id="findings" aria-labelledby="findings-heading">
  <div class="section-heading report-section-heading">
    <div>
      <p class="eyebrow">Benchmark-design findings</p>
      <h2 id="findings-heading">The evaluator is part of the experiment.</h2>
    </div>
    <p>Optimization exposed scientific structure, verifier mistakes, allocation assumptions, and one genuine specification loophole.</p>
  </div>
  <div class="finding-grid">{findings}</div>
</section>
"""


def index_page(
    site: dict[str, Any], report: dict[str, Any], tasks: list[dict[str, Any]]
) -> str:
    docker = sum(task["status"]["kind"] in {"validated", "caveat"} for task in tasks)
    qualified = sum(task["status"]["kind"] == "qualified" for task in tasks)
    provisional = sum(task["status"]["kind"] == "provisional" for task in tasks)
    special = sum(task["status"]["kind"] in {"caveat", "feasibility"} for task in tasks)
    rows = []
    cards = []
    for task in tasks:
        metric = task["metric"]
        rows.append(
            f"""
<tr data-task-kind="{escape(task["status"]["kind"])}">
  <td class="task-cell">
    <a href="{task_href('', task)}">
      <span class="task-number">{escape(task["id"])}</span>
      <span><strong>{escape(task["title"])}</strong><small>{escape(task["summary"])}</small></span>
    </a>
  </td>
  <td>{status_badge(task)}</td>
  <td class="metric-cell"><span>{escape(metric["baseline"])}</span><small>reference mean</small></td>
  <td class="metric-cell"><span>{escape(metric["candidate"])}</span><small>optimized mean</small></td>
  <td class="improvement-cell"><strong>{escape(metric["headline"])}</strong><small>{escape(metric["improvement"])}</small></td>
  <td class="row-arrow" aria-hidden="true">→</td>
</tr>
"""
        )
        cards.append(
            f"""
<a class="task-card-mobile" href="{task_href('', task)}" data-task-kind="{escape(task["status"]["kind"])}">
  <div class="task-card-top"><span class="task-number">{escape(task["id"])}</span>{status_badge(task)}</div>
  <h3>{escape(task["title"])}</h3>
  <p>{escape(task["summary"])}</p>
  <div class="task-card-metrics">
    <span><small>Original</small><strong>{escape(metric["baseline"])}</strong></span>
    <span><small>Optimized</small><strong>{escape(metric["candidate"])}</strong></span>
    <span class="task-card-improvement"><small>Result</small><strong>{escape(metric["headline"])}</strong></span>
  </div>
</a>
"""
        )

    return (
        common_head(
            site=site,
            title=f'{site["title"]} — Results',
            description=site["description"],
            canonical_path="",
            root_prefix="",
        )
        + f"""
<body class="landing-page" data-page="results">
{header(site, "")}
<main id="main-content">
  <section class="landing-hero">
    <div class="hero-grid" aria-hidden="true"></div>
    <div class="hero-copy">
      <p class="eyebrow"><span class="live-dot"></span> Final public report · updated {escape(site["updated"])}</p>
      <h1>{escape(report["title"])}</h1>
      <p class="hero-description">{escape(report["lede"])}</p>
      <div class="hero-actions">
        <a class="button button-primary" href="#figures">Explore the figures <span aria-hidden="true">↓</span></a>
        <a class="button button-secondary" href="#task-index">Open task evidence</a>
      </div>
    </div>
    <aside class="hero-ledger report-ledger" aria-label="Final report at a glance">
      <div class="ledger-header"><span>At a glance</span><small>three evidence axes</small></div>
      <div class="report-ledger-grid">
        <div><span>Fable 5</span><strong>12/12</strong><small>hybrid protocol</small></div>
        <div><span>GPT-5.6 high</span><strong>10/12</strong><small>final validity</small></div>
        <div><span>GPT-5.6 ultra</span><strong>11/12†</strong><small>after adjudication</small></div>
        <div><span>Expert + AI</span><strong>2.88×</strong><small>ordinary-task GM</small></div>
      </div>
      <p>Validity, generated-artifact efficiency, and expert co-optimization use different denominators. Read each result in its own evidence frame.</p>
    </aside>
  </section>

  {outcome_cards(report)}
  {evidence_model(report)}
  {graph_gallery(report)}

  <section class="task-index section-shell" id="task-index">
    <div class="section-heading report-section-heading">
      <div>
        <p class="eyebrow">Expert co-optimization index</p>
        <h2>Every task, its result and boundary.</h2>
      </div>
      <p>Open a task for the scientific problem, the decisive implementation insight, the focused git diff, and its public provenance.</p>
    </div>
    <div class="task-filter-row">
      <div class="filter-tabs" role="group" aria-label="Filter task evidence">
        <button type="button" class="is-active" data-filter="all" aria-pressed="true">All <span>12</span></button>
        <button type="button" data-filter="docker" aria-pressed="false">6-pair Docker <span>{docker}</span></button>
        <button type="button" data-filter="qualified" aria-pressed="false">5-pair <span>{qualified}</span></button>
        <button type="button" data-filter="local" aria-pressed="false">Local <span>{provisional}</span></button>
        <button type="button" data-filter="special" aria-pressed="false">Special <span>{special}</span></button>
      </div>
      <p class="filter-count" aria-live="polite"><strong data-filter-count>12</strong> tasks shown</p>
    </div>

    <div class="evidence-note">
      <span class="note-icon" aria-hidden="true">i</span>
      <p><strong>Within-task evidence, not a pooled runtime leaderboard.</strong> The 2.88× summary is a descriptive geometric mean of ten ordinary same-task ratios. Task 07's design reduction and Task 08's feasibility result are excluded. <span>† local-engine or adjudicated evidence; ‡ five-pair estimate.</span></p>
    </div>

    <div class="task-table-wrap">
      <table class="task-table">
        <caption class="sr-only">Task-by-task expert and optimized runtime evidence</caption>
        <thead><tr><th scope="col">Task</th><th scope="col">Status</th><th scope="col">Original</th><th scope="col">Optimized</th><th scope="col">Mean paired result</th><th scope="col"><span class="sr-only">Open</span></th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <div class="task-cards-mobile">{''.join(cards)}</div>

    <div class="status-legend" aria-label="Status legend">
      <span><i class="status-validated"></i> Six-pair Docker</span>
      <span><i class="status-qualified"></i> Five-pair qualified</span>
      <span><i class="status-provisional"></i> Six-pair local</span>
      <span><i class="status-caveat"></i> Design reduction</span>
      <span><i class="status-feasibility"></i> Feasibility</span>
    </div>
  </section>

  {report_findings(report)}
</main>
{footer(site, "")}
{search_dialog(tasks, "")}
<script src="assets/app.js" defer></script>
</body>
</html>
"""
    )


def task_page(
    site: dict[str, Any],
    tasks: list[dict[str, Any]],
    task: dict[str, Any],
    patch_excerpt: str,
) -> str:
    root_prefix = "../../"
    metric = task["metric"]
    task_index = tasks.index(task)
    previous_task = tasks[task_index - 1] if task_index > 0 else None
    next_task = tasks[task_index + 1] if task_index < len(tasks) - 1 else None
    caveat = (
        f'<div class="callout callout-caveat"><strong>Evidence boundary</strong><p>{escape(task["caveat"])}</p></div>'
        if task.get("caveat")
        else ""
    )
    context = measurement_context(task)
    insight_cards = "".join(
        f'<article class="insight-card"><span>{index:02d}</span><h3>{escape(insight["title"])}</h3><p>{escape(insight["body"])}</p></article>'
        for index, insight in enumerate(task["insights"], start=1)
    )
    previous_link = (
        f'<a href="{task_href(root_prefix, previous_task)}"><small>Previous</small><span>← Task {escape(previous_task["id"])}</span><strong>{escape(previous_task["short_title"])}</strong></a>'
        if previous_task
        else "<span></span>"
    )
    next_link = (
        f'<a class="next" href="{task_href(root_prefix, next_task)}"><small>Next</small><span>Task {escape(next_task["id"])} →</span><strong>{escape(next_task["short_title"])}</strong></a>'
        if next_task
        else "<span></span>"
    )
    diff_id = f'diff-{task["slug"]}'
    reference_url = task.get("reference_url") or repo_file_url(
        site, task["reference_path"]
    )
    reference_label = task.get("reference_diff_path", task["reference_path"])
    candidate_url = task.get("candidate_url") or repo_file_url(
        site, task["candidate_path"]
    )
    evidence_url = task.get("evidence_url") or repo_file_url(
        site, task["evidence_path"]
    )
    if task.get("candidate_url"):
        source_note = "Final report candidate from the linked ORBIT-Q evidence PR"
    elif "/variants/" in task["candidate_path"]:
        source_note = "Tracked secondary variant"
    else:
        source_note = "Final report candidate source"

    return (
        common_head(
            site=site,
            title=f'Task {task["id"]}: {task["title"]} — {site["title"]}',
            description=f'{task["summary"]} Result: {metric["headline"]}.',
            canonical_path=f'tasks/{task["slug"]}/',
            root_prefix=root_prefix,
        )
        + f"""
<body class="docs-page" data-page="task" data-task="{escape(task["id"])}">
{header(site, root_prefix, docs=True)}
<div class="docs-grid">
  {sidebar(site, tasks, root_prefix, current=task["slug"])}
  <main class="docs-main" id="main-content">
    <article class="docs-article">
      <nav class="breadcrumbs" aria-label="Breadcrumb">
        <a href="{root_prefix}">Results</a><span>/</span><span>Task {escape(task["id"])}</span>
      </nav>
      <header class="task-header">
        <div class="task-title-row">
          <div>
            <p class="eyebrow">ORBIT-Q · Task {escape(task["id"])}</p>
            <h1>{escape(task["title"])}</h1>
            <p class="task-deck">{escape(task["summary"])}</p>
          </div>
          {status_badge(task)}
        </div>
      </header>

      <section class="result-hero" aria-labelledby="result-heading">
        <div class="result-headline">
          <p class="eyebrow" id="result-heading">Recorded result</p>
          <strong>{escape(metric["headline"])}</strong>
          <span>{escape(metric["improvement"])}</span>
        </div>
        <div class="result-visual">
          {metric_bar(task)}
        </div>
        <dl class="result-strip">
          <div><dt>Pairs</dt><dd>{escape(metric["pairs"])}</dd></div>
          <div><dt>Engine</dt><dd>{escape(metric["engine"])}</dd></div>
          <div><dt>95% paired CI</dt><dd>{escape(metric["ci"])}</dd></div>
          <div><dt>Pair wins</dt><dd>{escape(metric["wins"])}</dd></div>
        </dl>
      </section>
      {caveat}
      {context}

      <section class="docs-section" id="problem">
        <div class="section-anchor"><span>01</span><h2>The problem</h2></div>
        <div class="prose">
          <p>{escape(task["problem"][0])}</p>
          <p>{escape(task["problem"][1])}</p>
        </div>
      </section>

      <section class="docs-section" id="insight">
        <div class="section-anchor"><span>02</span><h2>What changed—and why</h2></div>
        <blockquote class="headline-insight"><span>Learning carried forward</span>{escape(task["headline_insight"])}</blockquote>
        <div class="insight-grid">{insight_cards}</div>
      </section>

      <section class="docs-section" id="diff">
        <div class="section-anchor"><span>03</span><h2>The important diff</h2></div>
        <p class="section-intro">A focused excerpt from the actual unified patch. Surrounding bookkeeping and unchanged implementation detail are omitted; the complete generated patch remains available below.</p>
        <div class="diff-shell">
          <div class="diff-toolbar">
            <div><span class="diff-dot"></span><strong>original → optimized</strong><small>{escape(source_note)}</small></div>
            <button type="button" class="copy-button" data-copy-target="{diff_id}">Copy diff</button>
          </div>
          <pre class="diff-block" tabindex="0"><code id="{diff_id}">{diff_markup(patch_excerpt)}</code></pre>
          <div class="diff-footer">
            <a href="{root_prefix}diffs/{escape(task["slug"])}.diff">View full patch</a>
            <span>{escape(reference_label)}</span>
          </div>
        </div>
      </section>

      <section class="docs-section" id="evidence">
        <div class="section-anchor"><span>04</span><h2>Evidence & provenance</h2></div>
        <div class="evidence-grid">
          <a href="{escape(evidence_url)}"><span>Campaign evidence</span><strong>{escape(task["evidence_path"])}</strong><small>Open the report on GitHub ↗</small></a>
          <a href="{escape(reference_url)}"><span>Immutable original</span><strong>{escape(reference_label)}</strong><small>Inspect source ↗</small></a>
          <a href="{escape(candidate_url)}"><span>Compared candidate</span><strong>{escape(task["candidate_path"])}</strong><small>Inspect source ↗</small></a>
        </div>
        <details class="method-details">
          <summary><span>How to interpret this result</span><small>measurement boundary</small></summary>
          <div>
            <p>Evaluator-reported runtime includes tracing and compilation initiated inside <code>run_solution(config)</code>. A runtime is eligible only when the unchanged evaluator reports <code>Overall: PASS</code>.</p>
            <p>Speedup intervals are scoped to matched pairs on one host and environment. Results are never pooled across heterogeneous tasks, and a recorded candidate is not a global hardware-independent SOTA claim.</p>
          </div>
        </details>
      </section>

      <nav class="page-pagination" aria-label="Adjacent tasks">{previous_link}{next_link}</nav>
    </article>
  </main>
  <aside class="docs-toc" aria-label="On this page">
    <p>On this page</p>
    <a href="#problem">The problem</a>
    <a href="#insight">What changed—and why</a>
    <a href="#diff">The important diff</a>
    <a href="#evidence">Evidence & provenance</a>
    <div class="toc-result"><span>Result</span><strong>{escape(metric["headline"])}</strong><small>{escape(task["status"]["label"])}</small></div>
  </aside>
</div>
{footer(site, root_prefix)}
{search_dialog(tasks, root_prefix)}
<script src="{root_prefix}assets/app.js" defer></script>
</body>
</html>
"""
    )


def methodology_page(site: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
    root_prefix = "../"
    return (
        common_head(
            site=site,
            title=f'Measurement protocol — {site["title"]}',
            description="How OrbitBreakers validates correctness, paired runtime, confidence, and promotion status.",
            canonical_path="methodology/",
            root_prefix=root_prefix,
        )
        + f"""
<body class="docs-page" data-page="methodology">
{header(site, root_prefix, docs=True)}
<div class="docs-grid">
  {sidebar(site, tasks, root_prefix, methodology_current=True)}
  <main class="docs-main" id="main-content">
    <article class="docs-article">
      <nav class="breadcrumbs" aria-label="Breadcrumb"><a href="{root_prefix}">Results</a><span>/</span><span>Protocol</span></nav>
      <header class="task-header methodology-header">
        <p class="eyebrow">Measurement contract</p>
        <h1>When a faster run becomes a result.</h1>
        <p class="task-deck">Correctness and framework fidelity are hard gates. Runtime evidence only becomes a promoted claim after matched, counterbalanced measurement.</p>
      </header>

      <section class="protocol-steps" aria-label="Promotion workflow">
        <article><span>01</span><h2>Hold the surface still</h2><p>Reference, evaluator, workload, environment, result parsing, and validity rules remain immutable during a solution experiment.</p></article>
        <article><span>02</span><h2>Pair the measurements</h2><p>Reference and candidate run in alternating order, in one task container, with a fresh evaluator process for every cell.</p></article>
        <article><span>03</span><h2>Promote on evidence</h2><p>At least six pairs pass; mean and median improve; the candidate wins at least 80%; the declared confidence interval stays above one.</p></article>
      </section>

      <section class="docs-section" id="status">
        <div class="section-anchor"><span>01</span><h2>Status vocabulary</h2></div>
        <div class="status-table-wrap">
          <table class="status-table">
            <thead><tr><th>Label</th><th>What it means</th><th>Allowed headline</th></tr></thead>
            <tbody>
              <tr><td><span class="status-badge status-validated"><span class="status-dot"></span>Docker promoted</span></td><td>Six or more matched passing pairs in the pinned Docker protocol and all promotion checks pass.</td><td>Measured speedup and runtime reduction.</td></tr>
              <tr><td><span class="status-badge status-qualified"><span class="status-dot"></span>Five-pair qualified</span></td><td>Five matched passing same-environment pairs support the final-report estimate; the engine is stated on the task page and the stricter six-pair promotion gate remains open.</td><td>Measured estimate, visibly marked ‡ and paired with its limitation.</td></tr>
              <tr><td><span class="status-badge status-provisional"><span class="status-dot"></span>Local six-pair</span></td><td>Six matched passing pairs under the pinned dependency lock, but Docker was unavailable.</td><td>Local-engine speedup, visibly marked †.</td></tr>
              <tr><td><span class="status-badge status-caveat"><span class="status-dot"></span>Design reduction</span></td><td>The executable task contract permits a shortcut that changes the scale of the measured computation.</td><td>Exact result plus a conservative full-workload fallback.</td></tr>
              <tr><td><span class="status-badge status-feasibility"><span class="status-dot"></span>OOM → PASS</span></td><td>The candidate restores a valid canonical result where the reference cannot return within the fixed allocation.</td><td>Feasibility, never an invented numerical ratio.</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="docs-section" id="math">
        <div class="section-anchor"><span>02</span><h2>Reported quantities</h2></div>
        <div class="formula-grid">
          <div><span>Runtime reduction</span><code>100 × (reference mean − candidate mean) / reference mean</code></div>
          <div><span>Paired speedup</span><code>reference runtimeᵢ / candidate runtimeᵢ</code></div>
        </div>
        <div class="callout"><strong>Descriptive aggregate only</strong><p>The 2.88× geometric mean summarizes ten ordinary within-task expert-optimization ratios. It excludes Task 07's design reduction and Task 08's feasibility result; raw runtimes across workloads are never pooled.</p></div>
      </section>

      <section class="docs-section" id="boundary">
        <div class="section-anchor"><span>03</span><h2>Public evaluation boundary</h2></div>
        <div class="prose">
          <p>Every workload configuration, seed, evaluator, and validity rule used by a campaign is a versioned public artifact. The evaluator-reported timer includes tracing and compilation that begins inside <code>run_solution(config)</code>.</p>
          <p>A failed, timed-out, mismatched, unpaired, or semantically invalid candidate has no runtime standing. Negative experiments remain useful as mechanism evidence, but they do not become a speedup.</p>
        </div>
        <div class="evidence-grid protocol-links">
          <a href="{escape(repo_file_url(site, "GOAL.md"))}"><span>Authority</span><strong>GOAL.md</strong><small>Read the complete promotion rule ↗</small></a>
          <a href="{escape(repo_file_url(site, "BENCHMARKING.md"))}"><span>Reproduction</span><strong>BENCHMARKING.md</strong><small>Run the harness ↗</small></a>
          <a href="{escape(repo_file_url(site, "datasets/public/manifest.json"))}"><span>Workloads</span><strong>Public manifest</strong><small>Inspect versioned cases ↗</small></a>
        </div>
      </section>
    </article>
  </main>
  <aside class="docs-toc" aria-label="On this page">
    <p>On this page</p>
    <a href="#status">Status vocabulary</a>
    <a href="#math">Reported quantities</a>
    <a href="#boundary">Public boundary</a>
  </aside>
</div>
{footer(site, root_prefix)}
{search_dialog(tasks, root_prefix)}
<script src="{root_prefix}assets/app.js" defer></script>
</body>
</html>
"""
    )


def not_found_page(site: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
    root_prefix = site_base_path(site)
    return (
        common_head(
            site=site,
            title=f'Page not found — {site["title"]}',
            description="The requested OrbitBreakers benchmark page does not exist.",
            canonical_path="404.html",
            root_prefix=root_prefix,
        )
        + f"""
<body class="not-found-page">
{header(site, root_prefix)}
<main class="not-found" id="main-content">
  <p class="eyebrow">404 · Route not found</p>
  <h1>This result is outside the public index.</h1>
  <p>Return to the twelve versioned tasks or inspect the repository directly.</p>
  <div class="hero-actions"><a class="button button-primary" href="{escape(root_prefix)}">Open result index</a><a class="button button-secondary" href="{escape(site["repository"])}">Open GitHub</a></div>
</main>
{footer(site, root_prefix)}
{search_dialog(tasks, root_prefix)}
<script src="{escape(root_prefix)}assets/app.js" defer></script>
</body>
</html>
"""
    )


def build(output: Path) -> None:
    site, report, tasks = load_data()
    output = output.resolve()
    forbidden = {ROOT.resolve(), Path.home().resolve(), Path("/").resolve()}
    if output in forbidden:
        raise ValueError(f"Refusing to replace unsafe output path: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    assets = output / "assets"
    shutil.copytree(STATIC_DIR, assets, ignore=shutil.ignore_patterns("og.png"))
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (output / "index.html").write_text(
        index_page(site, report, tasks), encoding="utf-8"
    )
    (output / "404.html").write_text(not_found_page(site, tasks), encoding="utf-8")

    methodology_dir = output / "methodology"
    methodology_dir.mkdir()
    (methodology_dir / "index.html").write_text(
        methodology_page(site, tasks), encoding="utf-8"
    )

    diffs_dir = output / "diffs"
    diffs_dir.mkdir()
    for task in tasks:
        patch = diff_patch(task)
        excerpt = focused_diff(task, patch)
        task_dir = output / "tasks" / task["slug"]
        task_dir.mkdir(parents=True)
        (task_dir / "index.html").write_text(
            task_page(site, tasks, task, excerpt), encoding="utf-8"
        )
        (diffs_dir / f'{task["slug"]}.diff').write_text(patch, encoding="utf-8")
        (diffs_dir / f'{task["slug"]}.focused.diff').write_text(
            excerpt, encoding="utf-8"
        )

    figures_dir = output / "figures"
    figures_dir.mkdir()
    figure_names = {
        figure[key]
        for tab in report["graph_tabs"]
        for figure in tab["figures"]
        for key in ("src", "svg")
        if figure.get(key)
    }
    for figure_name in sorted(figure_names):
        shutil.copy2(FIGURES_DIR / figure_name, figures_dir / figure_name)

    search_index = [
        {
            "id": task["id"],
            "title": task["title"],
            "summary": task["summary"],
            "status": task["status"]["label"],
            "result": task["metric"]["headline"],
            "insights": [insight["title"] for insight in task["insights"]],
            "url": f'tasks/{task["slug"]}/',
        }
        for task in tasks
    ]
    (output / "search-index.json").write_text(
        json.dumps(search_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sitemap_urls = [
        site["site_url"],
        f'{site["site_url"]}methodology/',
        *[f'{site["site_url"]}tasks/{task["slug"]}/' for task in tasks],
    ]
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(
            f"  <url><loc>{escape(url)}</loc><lastmod>{escape(site['updated'])}</lastmod></url>\n"
            for url in sitemap_urls
        )
        + "</urlset>\n"
    )
    (output / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    (output / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {site['site_url']}sitemap.xml\n",
        encoding="utf-8",
    )

    llms_lines = [
        f"# {site['title']}",
        "",
        site["description"],
        "",
        "## Final synthesis",
        "- Fable 5: 12/12 official final reward under a hybrid protocol; solver resources are not directly comparable with Harbor runs.",
        "- GPT-5.6 Sol: 10/12 at high effort and 11/12 at ultra after Task 07 source adjudication.",
        "- Expert + AI: 2.88x descriptive geometric mean across ten ordinary supported campaigns.",
        "- Task 07 is a 45.758x challenge-design reduction; Task 08 is OOM-to-PASS feasibility with no confirmed runtime speedup.",
        "",
        "## Pages",
        f"- [Result index]({site['site_url']})",
        f"- [Measurement protocol]({site['site_url']}methodology/)",
    ]
    llms_lines.extend(
        f"- [Task {task['id']}: {task['title']}]({site['site_url']}tasks/{task['slug']}/): {task['metric']['headline']} — {task['status']['label']}"
        for task in tasks
    )
    llms_lines.extend(
        [
            "",
            "## Source",
            f"- [GitHub repository]({site['repository']})",
            "- Claims are scoped to the public workload and recorded measurement environment.",
        ]
    )
    (output / "llms.txt").write_text("\n".join(llms_lines) + "\n", encoding="utf-8")

    og_source = SITE_DIR / "static" / "og.png"
    if og_source.exists():
        shutil.copy2(og_source, output / "og.png")

    print(
        f"Built {len(tasks)} task pages, {len(figure_names)} report figures, and supporting routes in "
        f"{output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_site",
        help="Static artifact directory (default: _site)",
    )
    args = parser.parse_args()
    build(args.output if args.output.is_absolute() else ROOT / args.output)


if __name__ == "__main__":
    main()
