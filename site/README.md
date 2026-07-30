# Benchmark results site

This directory builds the static GitHub Pages documentation for the twelve
ORBIT-Q expert benchmark tasks.

The site is deliberately dependency-free. Its Fumadocs-inspired shell,
responsive task index, search dialog, dark mode, and focused patch views are
generated from:

- `site/data/tasks.json` for reviewed claims and insight copy;
- `references/task-XX/` and `src/solutions/task-XX/` for unified diffs;
- `site/static/` for the shared visual and interaction layer.

Build and validate:

```bash
npm run check
```

The artifact is written to `_site/`. GitHub Actions deploys that directory
only from `main`; pull requests run the same build and static checks without
publishing.

When benchmark evidence changes, update the corresponding data record only
after its campaign report and promotion status are final. Keep provisional,
feasibility, and under-sampled results visibly distinct from six-pair Docker
claims.
