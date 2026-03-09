# MLflow Run Comparison UI

A lightweight Streamlit app for browsing and comparing MLflow runs from a remote tracking server. Built as a better alternative to the stock MLflow chart comparison view.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- VPN access to your MLflow tracking server
- MLflow credentials (username/password)

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

This creates a `.venv` and installs all pinned dependencies from `uv.lock`.

### 2. Set environment variables

The app authenticates to the MLflow tracking server using these environment variables. Set them in your shell before launching:

```bash
export MLFLOW_TRACKING_URI=https://your-tracking-server.example.com
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password
```

### 3. Launch the app

```bash
uv run streamlit run app.py -- --group-name "my-group" --experiment-name "my-experiment"
```

The `--group-name` and `--experiment-name` flags set the default experiment shown in the sidebar (formatted as `group/experiment`). Both are optional and fall back to placeholder values if omitted.

The app opens in your browser at `http://localhost:8501`.

## Usage

### Experiment selection

The sidebar has a text input for the experiment name. It defaults to the value passed via `--group-name` / `--experiment-name` at launch. Change it to any experiment on your tracking server.

### Filtering runs

- **Filter by run name** — substring match (case-insensitive) on the run name
- **Filter by tags** — substring match across all tag key=value pairs

### Selecting runs

- Click the checkbox next to individual runs in the table
- Use **Select all visible** / **Clear selection** buttons above the table
- Selections persist while you change filters or metrics

### Comparing metrics

- The **Metrics** multiselect is pre-populated with a default set of metrics
- Add or remove metrics freely — each selected metric gets its own chart tile
- Adjust **Charts per row** (1-4) in the sidebar to control the grid layout
- Each chart shows step vs. value with one line per selected run

### Exporting a filter for the MLflow UI

Click **Copy MLflow filter for selected runs** to generate a filter string like:

```
attributes.run_id IN ('abc123', 'def456')
```

Paste this into the MLflow UI search bar to view the same runs there.

### Refreshing data

Click **Refresh data** in the sidebar to clear the cache and re-fetch runs and metric histories from the server. Data is cached for 2 minutes by default.

## Project Structure

```
mlflow_compare_ui/
├── app.py              # Streamlit UI
├── mlflow_utils.py     # MLflow client helper functions
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Pinned dependency lockfile
└── README.md
```

## Configuration

The default experiment is set at launch via CLI flags (see [Launch the app](#3-launch-the-app) above).

To change the default metrics pre-selected in the UI, edit the `DEFAULT_METRICS` list at the top of `app.py`.
