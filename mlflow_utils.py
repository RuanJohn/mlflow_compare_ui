"""Helper functions for querying the MLflow tracking server."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

from cachetools import TTLCache
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

NOISY_TAG_PREFIXES = (
    "mlflow.source",
    "mlflow.user",
    "mlflow.log-model",
    "mlflow.docker",
    "mlflow.databricks",
)

_client: Optional[MlflowClient] = None
_client_lock = threading.Lock()

_runs_cache: TTLCache = TTLCache(maxsize=64, ttl=120)
_metric_cache: TTLCache = TTLCache(maxsize=2048, ttl=120)
_cache_lock = threading.Lock()

_executor = ThreadPoolExecutor(max_workers=16)


def get_client() -> MlflowClient:
    """Return a singleton MlflowClient (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MlflowClient()
    return _client


def clear_caches() -> None:
    """Flush all server-side caches."""
    with _cache_lock:
        _runs_cache.clear()
        _metric_cache.clear()


def get_experiment_by_name(name: str) -> Optional[dict[str, Any]]:
    """Return experiment metadata as a dict, or None."""
    client = get_client()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return None
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "lifecycle_stage": exp.lifecycle_stage,
    }


def list_runs(experiment_id: str) -> list[dict[str, Any]]:
    """Fetch all runs for an experiment. Results are TTL-cached for 120s."""
    with _cache_lock:
        cached = _runs_cache.get(experiment_id)
    if cached is not None:
        return cached

    client = get_client()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    records: list[dict[str, Any]] = []
    for r in runs:
        tags = dict(r.data.tags) if r.data.tags else {}
        run_name = tags.get("mlflow.runName") or r.info.run_name or r.info.run_id
        start_dt = datetime.fromtimestamp(r.info.start_time / 1000, tz=timezone.utc)
        records.append({
            "run_id": r.info.run_id,
            "run_name": run_name,
            "start_time": start_dt.isoformat(),
            "start_time_ms": r.info.start_time,
            "status": r.info.status,
            "tags": tags,
            "tags_list": tags_to_list(tags),
            "metric_keys": sorted(r.data.metrics.keys()) if r.data.metrics else [],
            "params": dict(r.data.params) if r.data.params else {},
        })

    with _cache_lock:
        _runs_cache[experiment_id] = records
    return records


def list_metric_names(runs: list[dict[str, Any]]) -> list[str]:
    """Collect the union of all metric keys across runs."""
    names: set[str] = set()
    for r in runs:
        names.update(r.get("metric_keys", []))
    return sorted(names)


def _fetch_one_history(
    run_id: str, metric_name: str, *, skip_cache: bool = False
) -> dict[str, Any]:
    """Fetch a single metric history, using cache unless skip_cache is set."""
    cache_key = (run_id, metric_name)
    if not skip_cache:
        with _cache_lock:
            cached = _metric_cache.get(cache_key)
        if cached is not None:
            return cached

    client = get_client()
    history = client.get_metric_history(run_id, metric_name)
    sorted_history = sorted(history, key=lambda m: m.step)
    result = {
        "run_id": run_id,
        "metric": metric_name,
        "steps": [m.step for m in sorted_history],
        "values": [m.value for m in sorted_history],
        "timestamps": [m.timestamp for m in sorted_history],
    }

    with _cache_lock:
        _metric_cache[cache_key] = result
    return result


def batch_metric_history(
    run_ids: list[str], metrics: list[str], *, skip_cache: bool = False
) -> list[dict[str, Any]]:
    """Fetch metric histories for all (run_id, metric) pairs in parallel."""
    pairs = [(rid, m) for rid in run_ids for m in metrics]
    if not pairs:
        return []

    futures = {
        _executor.submit(_fetch_one_history, rid, m, skip_cache=skip_cache): (rid, m)
        for rid, m in pairs
    }

    results: list[dict[str, Any]] = []
    for future in as_completed(futures):
        rid, m = futures[future]
        try:
            results.append(future.result())
        except Exception as exc:
            log.warning("Failed to fetch metric %s for run %s: %s", m, rid, exc)
            results.append({
                "run_id": rid,
                "metric": m,
                "steps": [],
                "values": [],
                "error": str(exc),
            })
    return results


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
