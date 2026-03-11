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

The app opens in your browser at `http://localhost:5050`.

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

### Run type toggle

Switch between **Training** and **Evaluation** in the sidebar to swap the default metric presets. Training mode pre-selects common RL training metrics; evaluation mode pre-selects evaluation/postprocessing metrics.

### Comparing metrics

- The metric selector is pre-populated with a default set based on the run type
- Search and check/uncheck metrics, or click the × on a pill to remove
- Click **Compare** to fetch and render charts
- Each selected metric gets its own chart tile (or grouped — see below)

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

### Exporting a filter for the MLflow UI

Click **Copy MLflow filter** in the sidebar to copy a filter string like:

```
attributes.run_id IN ('abc123', 'def456')
```

Paste this into the MLflow UI search bar to view the same runs there.

### Refreshing data

Click the **↻ Refresh** button in the top bar to clear the server-side cache and re-fetch everything. Data is cached for 2 minutes by default.

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

- **Backend** (`app.py` + `mlflow_utils.py`): API endpoints for experiment resolution, run listing, and batch metric history. Uses a singleton `MlflowClient`, `TTLCache` for 120s caching, `ThreadPoolExecutor` with 16 workers for parallel metric fetches, and `orjson` for fast JSON serialization.
- **Frontend** (`templates/index.html`): All UI state and filtering lives in the browser. Charts render with uPlot (~35KB, canvas-based). No build step required.

## Configuration

The default experiment is set at launch via CLI flags (see [Launch the app](#3-launch-the-app) above).

Default metric presets for each run type are defined in the `DEFAULT_METRICS_TRAINING` and `DEFAULT_METRICS_EVALUATION` arrays in `templates/index.html`.
