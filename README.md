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

The `--group-name` and `--experiment-name` flags set the default experiment shown in the top bar (formatted as `group/experiment`). Both are optional and fall back to placeholder values if omitted.

Additional flags:

- `--port` — port to listen on (default: 5050)
- `--host` — host to bind to (default: 127.0.0.1)

The app opens in your browser at `http://localhost:5050`. The default experiment's run data is prefetched in the background on startup, so the first page load is typically fast.

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

Below the selected runs panel, a parameters comparison table appears automatically when runs are selected. It displays all run parameters in a scrollable table with:

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

Click the **↻ Refresh** button in the top bar to clear the server-side cache and re-fetch everything. Run metadata is cached for 2 minutes by default; auto-refresh bypasses the cache for metric data only.

## Project Structure

```
mlflow_compare_ui/
├── app.py              # Flask backend with API endpoints
├── mlflow_utils.py     # MLflow client helpers, caching, parallel fetcher
├── templates/
│   └── index.html      # Single-page frontend (vanilla JS + uPlot)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Pinned dependency lockfile
└── README.md
```

## Architecture

The app is a Flask backend serving a single-page vanilla JS frontend:

- **Backend** (`app.py` + `mlflow_utils.py`): API endpoints for experiment resolution, run listing (including parameters), and batch metric history. Uses a singleton `MlflowClient`, `TTLCache` for 120s caching, `ThreadPoolExecutor` with 16 workers for parallel metric fetches, and `orjson` for fast JSON serialization. The default experiment is prefetched in a background thread at startup to warm the cache.
- **Frontend** (`templates/index.html`): All UI state and filtering lives in the browser. Charts render with uPlot (~35KB, canvas-based). No build step required. Metric data for active charts is auto-refreshed every 30 seconds.

## Configuration

The default experiment is set at launch via CLI flags (see [Launch the app](#3-launch-the-app) above).

Default metric presets for each run type are defined in the `DEFAULT_METRICS_TRAINING` and `DEFAULT_METRICS_EVALUATION` arrays in `templates/index.html`.
