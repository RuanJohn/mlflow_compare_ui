"""Helper functions for querying the MLflow tracking server."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

NOISY_TAG_PREFIXES = (
    "mlflow.source",
    "mlflow.user",
    "mlflow.log-model",
    "mlflow.docker",
    "mlflow.databricks",
)


def get_client() -> MlflowClient:
    return MlflowClient()


def get_experiment_by_name(name: str):
    """Return an Experiment object or None."""
    client = get_client()
    return client.get_experiment_by_name(name)


@st.cache_data(ttl=120, show_spinner=False)
def list_runs(experiment_id: str) -> pd.DataFrame:
    """Fetch all runs for an experiment and return a tidy DataFrame."""
    client = get_client()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    if not runs:
        return pd.DataFrame()

    records = []
    for r in runs:
        tags = dict(r.data.tags) if r.data.tags else {}
        run_name = tags.get("mlflow.runName") or r.info.run_name or r.info.run_id
        records.append(
            {
                "run_id": r.info.run_id,
                "run_name": run_name,
                "start_time": pd.Timestamp(r.info.start_time, unit="ms", tz="UTC").tz_convert("US/Eastern"),
                "status": r.info.status,
                "tags": tags,
                "tags_list": tags_to_list(tags),
                "metrics": dict(r.data.metrics) if r.data.metrics else {},
            }
        )
    return pd.DataFrame(records)


def list_metric_names(runs_df: pd.DataFrame) -> list[str]:
    """Collect the union of all metric keys across runs."""
    names: set[str] = set()
    for m in runs_df["metrics"]:
        names.update(m.keys())
    return sorted(names)


@st.cache_data(ttl=120, show_spinner=False)
def get_metric_history_df(run_id: str, metric_name: str) -> pd.DataFrame:
    """Return a DataFrame with columns [step, value] for one metric."""
    client = get_client()
    history = client.get_metric_history(run_id, metric_name)
    if not history:
        return pd.DataFrame(columns=["step", "value"])
    return pd.DataFrame([{"step": m.step, "value": m.value} for m in history])


def tags_to_list(tags: dict, exclude_noisy: bool = True) -> list[str]:
    """Return a list of `key=value` strings suitable for pill/chip display."""
    pairs: list[str] = []
    for k, v in sorted(tags.items()):
        if exclude_noisy and any(k.startswith(p) for p in NOISY_TAG_PREFIXES):
            continue
        if k == "mlflow.runName":
            continue
        pairs.append(f"{k}={v}")
    return pairs
