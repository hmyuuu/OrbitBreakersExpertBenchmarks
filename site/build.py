#!/usr/bin/env python3
"""Build the dependency-free OrbitBreakers GitHub Pages artifact."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
DATA_PATH = SITE_DIR / "data" / "tasks.json"
STATIC_DIR = SITE_DIR / "static"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
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
    return payload["site"], tasks


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
<header class="site-header">
  <div class="header-inner">
    <div class="header-leading">
      {mobile_button}
      <a class="brand" href="{root_prefix}">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <span class="brand-copy"><strong>OrbitBreakers</strong><small>Benchmark results</small></span>
      </a>
    </div>
    <nav class="header-nav" aria-label="Primary">
      <a href="{root_prefix}" data-nav="results">Results</a>
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
      <input id="task-search" type="search" placeholder="Search a task, method, or result…" autocomplete="off">
      <button type="button" class="search-close" data-close-search aria-label="Close search">Esc</button>
    </div>
    <p class="search-label" id="search-title">Benchmark tasks</p>
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
    reference = ROOT / task["reference_path"]
    candidate = ROOT / task["candidate_path"]
    before = reference.read_text(encoding="utf-8").splitlines(keepends=True)
    after = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f'a/{task["reference_path"]}',
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
  <div class="runtime-row"><span>Campaign best</span><div class="runtime-track"><i class="bar-candidate" style="width: 42%"></i></div><strong>{escape(metric["candidate"])}</strong></div>
</div>
"""
    percentage = max(2.0, min(100.0, 100.0 * candidate_value / baseline_value))
    return f"""
<div class="runtime-bars" aria-label="Mean runtime comparison">
  <div class="runtime-row"><span>Original</span><div class="runtime-track"><i class="bar-reference" style="width: 100%"></i></div><strong>{escape(metric["baseline"])}</strong></div>
  <div class="runtime-row"><span>Campaign best</span><div class="runtime-track"><i class="bar-candidate" style="width: {percentage:.2f}%"></i></div><strong>{escape(metric["candidate"])}</strong></div>
</div>
"""


def index_page(site: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
    validated = sum(task["status"]["kind"] in {"validated", "caveat"} for task in tasks)
    provisional = sum(task["status"]["kind"] == "provisional" for task in tasks)
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
  <td class="metric-cell"><span>{escape(metric["candidate"])}</span><small>campaign best</small></td>
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
    <span><small>Best</small><strong>{escape(metric["candidate"])}</strong></span>
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
<main>
  <section class="landing-hero">
    <div class="hero-grid" aria-hidden="true"></div>
    <div class="hero-copy">
      <p class="eyebrow"><span class="live-dot"></span> Public benchmark record · updated {escape(site["updated"])}</p>
      <h1>Break the runtime.<br><em>Keep the contract.</em></h1>
      <p class="hero-description">Twelve TensorCircuit-NG expert solutions, measured as paired experiments. Every task page leads with the problem, the evidence, and the focused source diff behind the result.</p>
      <div class="hero-actions">
        <a class="button button-primary" href="#task-index">Explore all tasks <span aria-hidden="true">↓</span></a>
        <a class="button button-secondary" href="methodology/">Read the protocol</a>
      </div>
    </div>
    <aside class="hero-ledger" aria-label="Benchmark status summary">
      <div class="ledger-header"><span>Evidence ledger</span><small>public / reproducible</small></div>
      <div class="ledger-stat ledger-stat-wide"><strong>12</strong><span>tasks indexed</span></div>
      <div class="ledger-stat"><strong>{validated}</strong><span>six-pair Docker results</span></div>
      <div class="ledger-stat"><strong>{provisional}</strong><span>six-pair local results</span></div>
      <div class="ledger-line"><span>Memory result</span><strong>OOM → PASS</strong></div>
      <div class="ledger-line"><span>Open measurement gate</span><strong>Task 06 · pair 6</strong></div>
      <p>Speedups are never pooled across tasks. A failed or under-sampled result stays visible without becoming a claim.</p>
    </aside>
  </section>

  <section class="task-index section-shell" id="task-index">
    <div class="section-heading">
      <div>
        <p class="eyebrow">Optimization index</p>
        <h2>Every task, one evidence row.</h2>
      </div>
      <div class="filter-tabs" role="group" aria-label="Filter tasks">
        <button type="button" class="is-active" data-filter="all">All <span>12</span></button>
        <button type="button" data-filter="validated">Docker <span>{validated}</span></button>
        <button type="button" data-filter="provisional">Local <span>{provisional}</span></button>
        <button type="button" data-filter="open">Open <span>2</span></button>
      </div>
    </div>

    <div class="evidence-note">
      <span class="note-icon" aria-hidden="true">i</span>
      <p><strong>Repository SOTA, not a global SOTA claim.</strong> Results compare the immutable bundled expert with the best recorded candidate on the public evaluator and stated environment. <span>† local-engine evidence; ‡ five-pair descriptive timing.</span></p>
    </div>

    <div class="task-table-wrap">
      <table class="task-table">
        <thead><tr><th>Task</th><th>Status</th><th>Original</th><th>Campaign best</th><th>Improvement</th><th><span class="sr-only">Open</span></th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <div class="task-cards-mobile">{''.join(cards)}</div>

    <div class="status-legend" aria-label="Status legend">
      <span><i class="status-validated"></i> Six-pair Docker</span>
      <span><i class="status-provisional"></i> Six-pair local</span>
      <span><i class="status-feasibility"></i> Feasibility result</span>
      <span><i class="status-pending"></i> Measurement open</span>
    </div>
  </section>

  <section class="principles section-shell">
    <div>
      <p class="eyebrow">Reading the results</p>
      <h2>The diff is part of the evidence.</h2>
    </div>
    <div class="principle-grid">
      <article><span>01</span><h3>Problem first</h3><p>Each page states the scientific workload and the computation that cannot be removed.</p></article>
      <article><span>02</span><h3>Claims are gated</h3><p>Correctness, matched pairs, engine, wins, and confidence interval remain visible beside the headline.</p></article>
      <article><span>03</span><h3>Mechanism over slogan</h3><p>A focused unified diff connects the runtime movement to the retained algorithmic insight.</p></article>
    </div>
  </section>
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
    reference_url = repo_file_url(site, task["reference_path"])
    candidate_url = repo_file_url(site, task["candidate_path"])
    evidence_url = repo_file_url(site, task["evidence_path"])
    source_note = (
        "Tracked repository-best variant"
        if "/variants/" in task["candidate_path"]
        else "Current campaign-best source"
    )

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
  <main class="docs-main">
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
            <div><span class="diff-dot"></span><strong>original → campaign best</strong><small>{escape(source_note)}</small></div>
            <button type="button" class="copy-button" data-copy-target="{diff_id}">Copy diff</button>
          </div>
          <pre class="diff-block" tabindex="0"><code id="{diff_id}">{diff_markup(patch_excerpt)}</code></pre>
          <div class="diff-footer">
            <a href="{root_prefix}diffs/{escape(task["slug"])}.diff">View full patch</a>
            <span>{escape(task["reference_path"])}</span>
          </div>
        </div>
      </section>

      <section class="docs-section" id="evidence">
        <div class="section-anchor"><span>04</span><h2>Evidence & provenance</h2></div>
        <div class="evidence-grid">
          <a href="{escape(evidence_url)}"><span>Campaign evidence</span><strong>{escape(task["evidence_path"])}</strong><small>Open the report on GitHub ↗</small></a>
          <a href="{escape(reference_url)}"><span>Immutable original</span><strong>{escape(task["reference_path"])}</strong><small>Inspect source ↗</small></a>
          <a href="{escape(candidate_url)}"><span>Compared candidate</span><strong>{escape(task["candidate_path"])}</strong><small>Inspect source ↗</small></a>
        </div>
        <details class="method-details">
          <summary><span>How to interpret this result</span><small>measurement boundary</small></summary>
          <div>
            <p>Evaluator-reported runtime includes tracing and compilation initiated inside <code>run_solution(config)</code>. A runtime is eligible only when the unchanged evaluator reports <code>Overall: PASS</code>.</p>
            <p>Speedup intervals are scoped to matched pairs on one host and environment. Results are never pooled across heterogeneous tasks, and “campaign best” is not a global hardware-independent SOTA claim.</p>
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
  <main class="docs-main">
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
              <tr><td><span class="status-badge status-provisional"><span class="status-dot"></span>Local six-pair</span></td><td>Six matched passing pairs under the pinned dependency lock, but Docker was unavailable.</td><td>Local-engine speedup, visibly marked †.</td></tr>
              <tr><td><span class="status-badge status-feasibility"><span class="status-dot"></span>OOM → PASS</span></td><td>The candidate restores a valid canonical result where the reference cannot return within the fixed allocation.</td><td>Feasibility, never an invented numerical ratio.</td></tr>
              <tr><td><span class="status-badge status-pending"><span class="status-dot"></span>One pair short</span></td><td>Correct directional evidence exists, but the minimum sample count is not met.</td><td>Pending, with descriptive timing only.</td></tr>
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
        <div class="callout"><strong>No aggregate leaderboard</strong><p>Runtime ratios across different scientific workloads are not additive or directly comparable. The landing page is an index, not a pooled benchmark score.</p></div>
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
    return (
        common_head(
            site=site,
            title=f'Page not found — {site["title"]}',
            description="The requested OrbitBreakers benchmark page does not exist.",
            canonical_path="404.html",
            root_prefix="",
        )
        + f"""
<body class="not-found-page">
{header(site, "")}
<main class="not-found">
  <p class="eyebrow">404 · Route not found</p>
  <h1>This result is outside the public index.</h1>
  <p>Return to the twelve versioned tasks or inspect the repository directly.</p>
  <div class="hero-actions"><a class="button button-primary" href="./">Open result index</a><a class="button button-secondary" href="{escape(site["repository"])}">Open GitHub</a></div>
</main>
{footer(site, "")}
{search_dialog(tasks, "")}
<script src="assets/app.js" defer></script>
</body>
</html>
"""
    )


def build(output: Path) -> None:
    site, tasks = load_data()
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
    (output / "index.html").write_text(index_page(site, tasks), encoding="utf-8")
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
        f"Built {len(tasks)} task pages and supporting routes in "
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
