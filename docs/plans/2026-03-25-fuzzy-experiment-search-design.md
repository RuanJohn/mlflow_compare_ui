# Fuzzy Experiment Search Design

**Date:** 2026-03-25
**Branch:** feat/scroll-fix-and-multi-experiment
**Approach:** Standalone implementation scoped to experiment input (Approach C)

## Decisions

| Aspect | Decision |
|--------|----------|
| Backend | New `GET /api/experiments` returning name list |
| Fetch strategy | Hybrid — backend fetches from MLflow, client caches in-memory. Fetch on page load + refresh every 10 min |
| Matching | Case-insensitive substring (`includes()`) |
| UI | Dropdown list (max 10 results) + ghost text overlay on top match |
| Keyboard | Arrow keys navigate, Tab accepts ghost text, Enter adds experiment, Escape closes |
| Scope | Self-contained, scoped to `#expInput` only — no shared abstractions |
| Display | Experiment name only (no metadata) |
| Loaded experiments | Excluded from suggestions |

## Backend

New GET endpoint in `app.py`:

```
GET /api/experiments → { "experiments": ["group1/exp-a", "group1/exp-b", ...] }
```

Calls `mlflow_client.search_experiments()` and returns a flat list of experiment names. No server-side caching — client manages refresh timing.

## Client: Data Fetching

- On page load, fetch `/api/experiments` and store in `allExperimentNames` (JS array)
- `setInterval` at 10 minutes to re-fetch and update the list
- In-memory only, no localStorage

## Client: UI

### HTML structure

- Wrap `#expInput` in a position-relative container
- Overlay `<span>` on the input (same font/padding, reduced opacity) for ghost text of top match
- Absolutely-positioned `<div>` below input for dropdown list (same visual pattern as `#paramSearchResults`)

### Behavior

- **On `input` event:** filter `allExperimentNames` with case-insensitive `includes()`, show top 10 matches in dropdown, set ghost text to first match
- **Arrow up/down:** navigate dropdown items, update ghost text to highlighted item
- **Tab:** accept ghost text into input
- **Enter:** if dropdown open, accept highlighted item and trigger add experiment; if closed, trigger add as today
- **Escape / blur:** close dropdown
- Already-loaded experiments are filtered out of suggestions

### Styling

- Dropdown uses existing dark theme CSS variables (`--bg`, `--fg`, `--accent`)
- Highlighted item gets accent background
- Ghost text shows input value + remaining suffix of match at reduced opacity
