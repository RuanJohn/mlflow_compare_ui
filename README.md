# MLflow Run Comparison UI

A fast Flask + vanilla JS app for browsing and comparing MLflow training runs. Built as a snappier alternative to the stock MLflow chart comparison view — all filtering happens client-side, metric histories are fetched in parallel, and charts render with uPlot.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- VPN access to your MLflow tracking server
- MLflow credentials (username/password or token)

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

This creates a `.venv` and installs all pinned dependencies from `uv.lock`.

### 2. Set environment variables

The app authenticates to the MLflow tracking server using these environment variables. Set them in your shell or source a `.env` file before launching:

```bash
export MLFLOW_TRACKING_URI=https://your-tracking-server.example.com
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password
```

Optional variables:

- `MLFLOW_TRACKING_TOKEN` — bearer token auth (used instead of username/password if set)
- `MLFLOW_TRACKING_INSECURE_TLS` — set to `true` to skip TLS verification

### 3. Launch the app

```bash
uv run python app.py --group-name "my-group" --experiment-name "my-experiment"
```

The app opens in your browser at `http://localhost:5050`. On first launch, all runs are fetched from MLflow and stored in a local SQLite cache. Subsequent launches serve cached data instantly and only fetch new runs incrementally.

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--group-name` | `your-group` | Default experiment group name, pre-filled in the UI as `group/experiment` |
| `--experiment-name` | `some-exp` | Default experiment name, pre-filled in the UI as `group/experiment` |
| `--port` | `5050` | Port to listen on |
| `--host` | `127.0.0.1` | Host to bind to |
| `--import-json FILE` | — | Path to an exported `.json` view file to auto-restore on startup (experiment, selected runs, metrics, chart settings) |
| `--cache-dir` | `~/.mlflow_compare_ui` | Directory for the persistent SQLite cache database |

## Usage

### Experiment selection

The top bar has a text input for the experiment name. It defaults to the value passed via `--group-name` / `--experiment-name` at launch. Press Enter or click **Load** to switch experiments.

### Filtering runs

Filters in the sidebar operate client-side for instant results:

- **Filter by run name** — substring match (case-insensitive) on the run name
- **Filter by tags** — substring match across all tag key=value pairs

### Selecting runs

- Click the checkbox next to individual runs in the table, or use the header checkbox to toggle all visible
- Use **Select all visible** / **Clear selection** in the sidebar
- Selections persist while you change filters or metrics

### Selected runs panel

When runs are selected, a panel appears below the table showing each run with:

- A colored dot matching its chart line color
- An editable legend name (defaults to the run name — click to rename)
- An × button to deselect

Custom legend names are used in chart legends and persist across re-renders.

### Parameters comparison

Parameters are loaded on demand — they are only fetched from MLflow when you first select a run (a spinner is shown while loading). Once fetched, params are cached in SQLite so they load instantly on subsequent visits.

The parameters table displays all run parameters in a scrollable table with:

- **Rows** = parameter names, **Columns** = selected runs (with colored dots matching chart colors)
- **Search and filter** — type in the search box and click **Add** to filter to specific parameters, shown as removable pills
- **Show diffs only** — toggle button that filters the table to only show parameters that differ between the selected runs, with differing cells highlighted
- **Sticky headers** — the parameter name column and header row stay visible while scrolling
- Long values are truncated with ellipsis; hover any cell to see the full value in a tooltip

### Run type toggle

Switch between **Training** and **Evaluation** in the sidebar to swap the default metric presets. Training mode pre-selects common RL training metrics; evaluation mode pre-selects evaluation/postprocessing metrics.

### Comparing metrics

- The metric selector is pre-populated with a default set based on the run type
- Search and check/uncheck metrics, or click the × on a pill to remove
- Click **Compare** to fetch and render charts
- Each selected metric gets its own chart tile (or grouped — see below)

### Auto-refresh

After clicking **Compare**, charts automatically refresh every 30 seconds by fetching fresh metric data (bypassing the cache) for the selected runs and metrics. A pulsing green indicator appears above the charts when auto-refresh is active. Polling stops when you load a new experiment, click Compare again, or clear the selection.

### Chart interactions

Charts use uPlot for fast rendering and support TensorBoard/MLflow-style interactions:

- **Drag to zoom** — click and drag a rectangle to zoom into a region
- **Scroll wheel zoom** — mouse wheel zooms the x-axis centered on the cursor
- **Double-click to reset** — resets zoom to the full data range
- **Reset Zoom button** — per-chart button in the top-right corner
- **Synced cursors** — hovering over one chart shows the crosshair on all others
- **Live legend** — legend values update as you hover

### Chart configuration

All chart settings are in the sidebar and take effect instantly (no re-fetch needed unless noted):

| Setting | Options |
|---|---|
| **Charts per row** | 1–4 (slider) |
| **X-Axis** | Step / Relative Wall Time |
| **Y-Axis Scale** | Linear / Log |
| **Smoothing** | EMA slider (0–0.99), like TensorBoard |
| **Chart Grouping** | One chart per metric / Auto-group by prefix / All on one chart |

### Sharing and exporting views

- **Share URL** — encodes the full view state (experiment, selected runs, metrics, settings) into a URL hash you can send to others
- **Export .json** — downloads the current view state as a JSON file
- **Import .json** — restores a previously exported view, including run selections and metric choices
- **Copy MLflow filter** — copies a filter string like `attributes.run_id IN ('abc123', 'def456')` for use in the MLflow UI search bar

### Refreshing data

Click the **↻ Refresh** button in the top bar to clear all caches (in-memory and SQLite) and re-fetch everything from MLflow.

## Project Structure

```
mlflow_compare_ui/
├── app.py              # Flask backend with API endpoints
├── mlflow_utils.py     # MLflow client helpers, caching, parallel fetcher
├── cache_db.py         # SQLite persistent cache (runs, params, metric histories)
├── templates/
│   └── index.html      # Single-page frontend (vanilla JS + uPlot)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Pinned dependency lockfile
└── README.md
```

## Architecture

The app is a Flask backend serving a single-page vanilla JS frontend, with a two-tier caching system for fast startup and low latency.

### Caching

Data is cached at two levels:

| Layer | Storage | Lifetime | Purpose |
|---|---|---|---|
| **L1** | In-memory `TTLCache` | 120 seconds | Avoids repeated SQLite reads within a session |
| **L2** | SQLite (`~/.mlflow_compare_ui/cache.db`) | Persistent until cleared | Survives app restarts; enables incremental sync |

The SQLite database stores four tables:

- **`runs`** — run metadata (id, name, start time, status, tags, metric key names)
- **`params`** — run parameters as key-value rows (lazy-loaded per run)
- **`metric_history`** — metric timeseries data as step/value/timestamp rows (lazy-loaded per run+metric)
- **`fetch_status`** — tracks which params and metrics have been fully fetched

### Incremental run sync

On startup (or when loading an experiment), `list_runs()` follows this flow:

1. Check the in-memory TTLCache — if a fresh result exists, return it immediately
2. Read all cached runs for the experiment from SQLite
3. Find the most recent `start_time_ms` in the cache
4. Query MLflow for only runs newer than that timestamp
5. Merge new runs into SQLite and return the combined result

On first launch (empty cache), all runs are fetched from MLflow. On subsequent launches, only new runs since the last fetch are pulled. Cached runs (including RUNNING ones) are served as-is for fast table rendering — fresh metric data is always fetched when you click Compare or via auto-refresh.

### Lazy loading

To keep the initial load fast, params and metric histories are not fetched until needed:

- **Params**: fetched from MLflow when a run is first selected in the UI, then cached in SQLite. Subsequent selections serve from cache.
- **Metric histories**: fetched from MLflow when **Compare** is clicked, then cached in SQLite. Subsequent chart loads for the same run+metric serve from cache. Auto-refresh bypasses the cache to pick up new data points.

Both use a `ThreadPoolExecutor` with 16 workers for parallel fetches.

### Staleness

- **Run metadata** — cached forever once fetched. New runs are picked up incrementally; status changes on existing runs are not re-checked (use Refresh to force a full re-fetch if needed).
- **Metric histories** — cached in SQLite on first fetch. Auto-refresh (every 30s) bypasses the cache and writes fresh data back, so the DB always has the latest values for actively viewed metrics.
- **Clear cache** — the Refresh button wipes both L1 and L2 caches and re-fetches everything from MLflow.

### Backend

`app.py` + `mlflow_utils.py` + `cache_db.py`: API endpoints for experiment resolution, run listing, param fetching, and batch metric history. Uses a singleton `MlflowClient`, `orjson` for fast JSON serialization, and SQLite in WAL mode for concurrent reads/writes. The default experiment is prefetched in a background thread at startup.

### Frontend

`templates/index.html`: All UI state and filtering lives in the browser. Charts render with uPlot (~35KB, canvas-based). No build step required.

## Configuration

The default experiment is set at launch via CLI flags (see [CLI Flags](#cli-flags) above).

Default metric presets for each run type are defined in the `DEFAULT_METRICS_TRAINING` and `DEFAULT_METRICS_EVALUATION` arrays in `templates/index.html`.
