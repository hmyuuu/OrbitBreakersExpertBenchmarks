# Benchmark results site

This directory builds the static GitHub Pages documentation for the twelve
ORBIT-Q expert benchmark tasks.

The site is deliberately dependency-free. Its Fumadocs-inspired shell,
responsive task index, search dialog, dark mode, and focused patch views are
generated from:

- the root `README.md` as the canonical final synthesis;
- `site/data/report.json` for report outcomes, evidence axes, graph tabs, and
  benchmark-design findings;
- `site/data/tasks.json` for reviewed claims and insight copy;
- `references/task-XX/` and `src/solutions/task-XX/` for unified diffs;
- the explicitly selected report charts in `docs/figures/`;
- `site/static/` for the shared visual and interaction layer.

Build and validate:

```bash
npm run check
```

The artifact is written to `_site/`. GitHub Actions deploys that directory
only from `main`; pull requests run the same build and static checks without
publishing.

When benchmark evidence changes, reconcile the final report, `report.json`,
and the corresponding task record before publishing. Keep six-pair Docker,
five-pair qualified, local, design-reduction, and feasibility results visibly
distinct.
