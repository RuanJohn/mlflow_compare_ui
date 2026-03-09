"""MLflow Run Comparison UI – a lightweight Streamlit app."""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from mlflow_utils import (
    get_experiment_by_name,
    get_metric_history_df,
    list_metric_names,
    list_runs,
)

parser = argparse.ArgumentParser(description="MLflow Run Comparison UI")
parser.add_argument("--group-name", default="your-group", help="Default experiment group name")
parser.add_argument("--experiment-name", default="some-exp", help="Default experiment name")
args = parser.parse_args(sys.argv[1:])

DEFAULT_EXPERIMENT_TO_SHOW = f"{args.group_name}/{args.experiment_name}"

DEFAULT_METRICS = [
    "actor/episode_return_mean",
    "actor/connected_ratio_mean",
    "actor/is_fully_connected_mean",
    "trainer/total_loss_mean",
    "trainer/clip_fraction_mean",
    "trainer/critic_stats/vf_loss_mean",
    "trainer/entropy_stats/policy_mean",
]

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MLflow Compare", layout="wide")
st.title("MLflow Run Comparison")

# ── session state defaults ───────────────────────────────────────────────────
if "selected_run_ids" not in st.session_state:
    st.session_state.selected_run_ids = set()

# ── sidebar: experiment selector ─────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment")
    experiment_name = st.text_input("Experiment name", value=DEFAULT_EXPERIMENT_TO_SHOW)

    if st.button("🔄 Refresh data"):
        list_runs.clear()
        get_metric_history_df.clear()
        st.rerun()

# ── resolve experiment ───────────────────────────────────────────────────────
experiment = get_experiment_by_name(experiment_name)
if experiment is None:
    st.error(f'Experiment **"{experiment_name}"** not found on the tracking server.')
    st.stop()

experiment_id = experiment.experiment_id
st.caption(f"Experiment ID: `{experiment_id}`")

# ── load runs ────────────────────────────────────────────────────────────────
with st.spinner("Loading runs…"):
    runs_df = list_runs(experiment_id)

if runs_df.empty:
    st.warning("No runs found for this experiment.")
    st.stop()

# ── sidebar: filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    name_filter = st.text_input("Filter by run name")
    tag_filter = st.text_input("Filter by tags")

filtered = runs_df.copy()
if name_filter:
    filtered = filtered[filtered["run_name"].str.contains(name_filter, case=False, na=False)]
if tag_filter:
    filtered = filtered[
        filtered["tags_list"].apply(
            lambda tags: any(tag_filter.lower() in t.lower() for t in tags)
        )
    ]

if filtered.empty:
    st.info("No runs match the current filters.")
    st.stop()

# ── select / clear actions ───────────────────────────────────────────────────
col_a, col_b, _ = st.columns([1, 1, 4])
with col_a:
    if st.button("Select all visible"):
        st.session_state.selected_run_ids = set(filtered["run_id"])
        st.rerun()
with col_b:
    if st.button("Clear selection"):
        st.session_state.selected_run_ids = set()
        st.rerun()

# ── run table ────────────────────────────────────────────────────────────────
st.subheader("Runs")

selected_ids: set[str] = st.session_state.selected_run_ids

display_rows: list[dict] = []
for _, row in filtered.iterrows():
    display_rows.append(
        {
            "selected": row["run_id"] in selected_ids,
            "run_name": row["run_name"],
            "start_time": row["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "status": row["status"],
            "run_id": row["run_id"],
            "tags": row["tags_list"],
        }
    )

display_df = pd.DataFrame(display_rows)

edited = st.data_editor(
    display_df,
    column_config={
        "selected": st.column_config.CheckboxColumn("✔", default=False, width=35),
        "run_name": st.column_config.TextColumn("Run Name", width=140),
        "start_time": st.column_config.TextColumn("Start Time", width=145),
        "status": st.column_config.TextColumn("Status", width=70),
        "run_id": st.column_config.TextColumn("Run ID", width=260),
        "tags": st.column_config.ListColumn("Tags", width=1500),
    },
    disabled=["run_name", "start_time", "status", "run_id", "tags"],
    hide_index=True,
    use_container_width=False,
    key="run_table",
)

new_selected: set[str] = set()
for idx, row in edited.iterrows():
    if row["selected"]:
        new_selected.add(row["run_id"])
st.session_state.selected_run_ids = new_selected

# ── MLflow filter export ─────────────────────────────────────────────────────
if new_selected:
    if st.button("📋 Copy MLflow filter for selected runs"):
        ids = ", ".join(f"'{rid}'" for rid in sorted(new_selected))
        mlflow_filter = f"attributes.run_id IN ({ids})"
        st.code(mlflow_filter, language="text")

# ── metric comparison ────────────────────────────────────────────────────────
st.subheader("Metric Comparison")

metric_names = list_metric_names(runs_df)
if not metric_names:
    st.info("No metrics recorded for these runs.")
    st.stop()

with st.sidebar:
    st.header("Charts")
    cols_per_row = st.slider("Charts per row", min_value=1, max_value=4, value=3)

default_selection = [m for m in DEFAULT_METRICS if m in metric_names]
selected_metrics = st.multiselect("Metrics", metric_names, default=default_selection)

selected_runs_df = filtered[filtered["run_id"].isin(st.session_state.selected_run_ids)]

if selected_runs_df.empty:
    st.info("Select one or more runs above to compare metrics.")
    st.stop()

if not selected_metrics:
    st.info("Select one or more metrics above to view charts.")
    st.stop()


# ── helper: build a single chart ─────────────────────────────────────────────
def _build_chart(metric: str, runs: pd.DataFrame) -> px.line | None:
    frames: list[pd.DataFrame] = []
    for _, run_row in runs.iterrows():
        hist = get_metric_history_df(run_row["run_id"], metric)
        if hist.empty:
            continue
        label = f"{run_row['run_name']}  ({run_row['start_time'].strftime('%m/%d %H:%M')})"
        hist = hist.assign(run=label)
        frames.append(hist)
    if not frames:
        return None
    chart_df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        chart_df,
        x="step",
        y="value",
        color="run",
        labels={"step": "Step", "value": metric, "run": "Run"},
        title=metric,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return fig


# ── tiled chart grid ─────────────────────────────────────────────────────────
with st.spinner("Loading metric histories…"):
    for i in range(0, len(selected_metrics), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(selected_metrics):
                break
            fig = _build_chart(selected_metrics[idx], selected_runs_df)
            if fig is None:
                col.warning(f"No data for **{selected_metrics[idx]}**")
            else:
                col.plotly_chart(fig, use_container_width=True)
