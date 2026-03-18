"""Helper functions for querying the MLflow tracking server."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

from cachetools import TTLCache
from mlflow.tracking import MlflowClient

import cache_db

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
    """Flush all in-memory and persistent caches."""
    with _cache_lock:
        _runs_cache.clear()
        _metric_cache.clear()
    cache_db.clear_cache()


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


def _mlflow_run_to_record(r: Any) -> dict[str, Any]:
    """Convert a single MLflow Run object to the dict format used by the app."""
    tags = dict(r.data.tags) if r.data.tags else {}
    run_name = tags.get("mlflow.runName") or r.info.run_name or r.info.run_id
    start_dt = datetime.fromtimestamp(r.info.start_time / 1000, tz=timezone.utc)
    return {
        "run_id": r.info.run_id,
        "run_name": run_name,
        "start_time": start_dt.isoformat(),
        "start_time_ms": r.info.start_time,
        "status": r.info.status,
        "tags": tags,
        "tags_list": tags_to_list(tags),
        "metric_keys": sorted(r.data.metrics.keys()) if r.data.metrics else [],
    }


def list_runs(experiment_id: str) -> list[dict[str, Any]]:
    """Fetch runs with incremental sync: SQLite cache -> MLflow for new/updated only.

    The in-memory TTLCache acts as L1 over the SQLite L2.
    """
    with _cache_lock:
        cached = _runs_cache.get(experiment_id)
    if cached is not None:
        return cached

    cached_runs = cache_db.get_cached_runs(experiment_id)
    max_ts = cache_db.get_max_start_time(experiment_id)

    client = get_client()
    new_records: list[dict[str, Any]] = []

    if max_ts is not None:
        filter_str = f"attributes.start_time > {max_ts}"
        new_mlflow_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_str,
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
        new_records = [_mlflow_run_to_record(r) for r in new_mlflow_runs]
    else:
        all_mlflow_runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
        new_records = [_mlflow_run_to_record(r) for r in all_mlflow_runs]
        cached_runs = []

    if new_records:
        cache_db.upsert_runs(new_records, experiment_id)

    existing_ids = {r["run_id"] for r in cached_runs}
    merged = list(cached_runs)
    for rec in new_records:
        if rec["run_id"] not in existing_ids:
            merged.append(rec)
            existing_ids.add(rec["run_id"])
        else:
            for i, c in enumerate(merged):
                if c["run_id"] == rec["run_id"]:
                    merged[i] = rec
                    break

    merged.sort(key=lambda r: r["start_time_ms"], reverse=True)

    with _cache_lock:
        _runs_cache[experiment_id] = merged
    return merged


def list_metric_names(runs: list[dict[str, Any]]) -> list[str]:
    """Collect the union of all metric keys across runs."""
    names: set[str] = set()
    for r in runs:
        names.update(r.get("metric_keys", []))
    return sorted(names)


def get_params_for_runs(run_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return params for the given runs, using SQLite cache with MLflow fallback."""
    if not run_ids:
        return {}

    cached = cache_db.get_cached_params(run_ids)
    missing_ids = [rid for rid in run_ids if rid not in cached]

    if missing_ids:
        client = get_client()
        futures = {
            _executor.submit(_fetch_params_for_run, client, rid): rid
            for rid in missing_ids
        }
        for future in as_completed(futures):
            rid = futures[future]
            try:
                params = future.result()
                cached[rid] = params
                cache_db.upsert_params(rid, params)
            except Exception as exc:
                log.warning("Failed to fetch params for run %s: %s", rid, exc)
                cached[rid] = {}

    return cached


def _fetch_params_for_run(client: MlflowClient, run_id: str) -> dict[str, str]:
    """Fetch params for a single run from MLflow."""
    run = client.get_run(run_id)
    return dict(run.data.params) if run.data.params else {}


def _fetch_one_history(
    run_id: str, metric_name: str, *, skip_cache: bool = False
) -> dict[str, Any]:
    """Fetch a single metric history with L1 (TTLCache) and L2 (SQLite) caching."""
    cache_key = (run_id, metric_name)

    if not skip_cache:
        with _cache_lock:
            mem_cached = _metric_cache.get(cache_key)
        if mem_cached is not None:
            return mem_cached

        db_cached = cache_db.get_cached_metric_history(run_id, metric_name)
        if db_cached is not None:
            with _cache_lock:
                _metric_cache[cache_key] = db_cached
            return db_cached

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

    cache_db.upsert_metric_history(
        run_id, metric_name, result["steps"], result["values"], result["timestamps"]
    )

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
