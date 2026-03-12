"""MLflow Run Comparison UI -- lightweight Flask app.

Usage:
    MLFLOW_TRACKING_URI=<uri> python app.py [--group-name G] [--experiment-name E]
    # Then open http://localhost:5050
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from base64 import b64encode
from typing import Any

import mlflow
import orjson
from flask import Flask, Response, render_template, request
from mlflow.exceptions import MlflowException

from mlflow_utils import (
    batch_metric_history,
    clear_caches,
    get_experiment_by_name,
    list_metric_names,
    list_runs,
)

log = logging.getLogger(__name__)

app = Flask(__name__)

_cli_defaults: dict[str, str] = {}


def json_response(data: Any, status: int = 200) -> Response:
    """Return a Response with orjson-serialised JSON body."""
    return Response(
        orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS),
        status=status,
        content_type="application/json",
    )


# ── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def handle_not_found(exc):
    return json_response({"error": "Not found"}, 404)


@app.errorhandler(Exception)
def handle_exception(exc):
    if isinstance(exc, MlflowException):
        msg = str(exc)
        if "403" in msg or "Forbidden" in msg:
            return json_response({
                "error": "MLflow returned 403 Forbidden. Check MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD."
            }, 403)
        if "401" in msg or "Unauthorized" in msg:
            return json_response({
                "error": "MLflow returned 401 Unauthorized. Check MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD."
            }, 401)
        return json_response({"error": msg}, 502)
    log.exception("Unhandled error")
    return json_response({"error": str(exc)}, 500)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        default_group=_cli_defaults.get("group_name", "your-group"),
        default_experiment=_cli_defaults.get("experiment_name", "some-exp"),
    )


@app.route("/api/experiment")
def api_experiment():
    name = request.args.get("name", "").strip()
    if not name:
        return json_response({"error": "Missing 'name' parameter"}, 400)
    result = get_experiment_by_name(name)
    if result is None:
        return json_response({"error": f"Experiment '{name}' not found"}, 404)
    return json_response(result)


@app.route("/api/runs")
def api_runs():
    experiment_id = request.args.get("experiment_id", "").strip()
    if not experiment_id:
        return json_response({"error": "Missing 'experiment_id' parameter"}, 400)
    runs = list_runs(experiment_id)
    return json_response({"runs": runs})


@app.route("/api/metric-names")
def api_metric_names():
    experiment_id = request.args.get("experiment_id", "").strip()
    if not experiment_id:
        return json_response({"error": "Missing 'experiment_id' parameter"}, 400)
    runs = list_runs(experiment_id)
    names = list_metric_names(runs)
    return json_response({"metrics": names})


@app.route("/api/metric-history", methods=["POST"])
def api_metric_history():
    body = request.get_json(silent=True) or {}
    run_ids = body.get("run_ids", [])
    metrics = body.get("metrics", [])
    if not run_ids or not metrics:
        return json_response(
            {"error": "Both 'run_ids' and 'metrics' are required"}, 400
        )
    skip_cache = bool(body.get("skip_cache", False))
    results = batch_metric_history(run_ids, metrics, skip_cache=skip_cache)
    return json_response({"results": results})


@app.route("/api/clear-cache", methods=["POST"])
def api_clear_cache():
    clear_caches()
    return json_response({"ok": True})


# ── Startup helpers ──────────────────────────────────────────────────────────

def check_connectivity() -> None:
    """Probe the MLflow server to find the API and test connectivity."""
    import requests

    uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
    insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in (
        "true", "1", "yes",
    )

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and password:
        encoded = b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    print("\nProbing MLflow API location:")

    probe_paths = [
        "/api/2.0/mlflow/experiments/search",
        "/mlflow/api/2.0/mlflow/experiments/search",
        "/ajax-api/2.0/mlflow/experiments/search",
    ]

    for path in probe_paths:
        url = f"{uri}{path}?max_results=1"
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=not insecure)
            print(f"  {path} -> {resp.status_code}")
            if resp.status_code == 200:
                print("  FOUND! API base is at this path.\n")
                return
        except Exception as exc:
            print(f"  {path} -> error: {exc}")

    try:
        resp = requests.get(
            uri, headers=headers, timeout=10, verify=not insecure, allow_redirects=False
        )
        print(f"\n  GET {uri} -> {resp.status_code}")
        if resp.status_code in (301, 302, 307, 308):
            print(f"  Redirects to: {resp.headers.get('Location', '?')}")
        else:
            print(f"  Content-Type: {resp.headers.get('Content-Type', '?')}")
            print(f"  Body (first 300 chars): {resp.text[:300]}")
    except Exception as exc:
        print(f"  GET {uri} -> error: {exc}")

    print("\n  Could not find MLflow API. The tracking URI might need a subpath.\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MLflow Run Comparison UI")
    parser.add_argument(
        "--group-name", default="your-group", help="Default experiment group name"
    )
    parser.add_argument(
        "--experiment-name", default="some-exp", help="Default experiment name"
    )
    parser.add_argument("--port", type=int, default=5050, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    _cli_defaults["group_name"] = args.group_name
    _cli_defaults["experiment_name"] = args.experiment_name

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print(
            "ERROR: MLFLOW_TRACKING_URI environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    uri = os.environ["MLFLOW_TRACKING_URI"]
    user = os.environ.get("MLFLOW_TRACKING_USERNAME", "(not set)")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN", "(not set)")
    print(f"MLflow server:   {uri}")
    print(f"MLflow user:     {user}")
    print(f"MLflow token:    {'set' if token != '(not set)' else '(not set)'}")
    print(
        f"MLflow password: {'set' if os.environ.get('MLFLOW_TRACKING_PASSWORD') else '(not set)'}"
    )

    insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in (
        "true", "1", "yes",
    )
    if insecure:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    uri = uri.rstrip("/")
    mlflow.set_tracking_uri(uri)

    check_connectivity()

    print(f"\n  Starting Flask on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
