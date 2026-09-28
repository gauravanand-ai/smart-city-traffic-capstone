#!/usr/bin/env python3
"""
deploy_api.py
Part 3, Task 6.3 (MLOps: deployment simulation).

A FastAPI mock-up of how the traffic-risk classifier built in Task 1 (and
versioned/registered in Task 6.1/6.2, model_registry.py) would actually be
served: a small HTTP API, not a script someone has to re-run for every new
reading. Accepts one raw traffic/weather reading and returns a high_risk
prediction and probability.

The model is loaded from MLflow's Model Registry at the "champion" alias
(models:/traffic-risk-classifier@champion) -- NOT from a hardcoded .joblib
path. This is the actual point of Task 6.1's registry work: promoting a new
model version to serving traffic is a one-line alias change in
model_registry.py (re-run it, or move the alias by hand), not a code change
or redeploy of this file.

Endpoints:
    GET  /health       -- liveness + which model version is currently loaded
    GET  /model-info    -- full details (params, metrics, run id) of the
                            currently-serving model version
    POST /predict/risk  -- the actual prediction endpoint

Run:
    python deploy_api.py [--tracking-uri URI] [--registered-name NAME]
                          [--alias champion] [--host 0.0.0.0] [--port 8000]
                          [--input PATH] [--log-file PATH] [-v]
    (starts a uvicorn server; Ctrl-C to stop)

Example request (see RESULTS_TASK6.md for the full worked examples):
    curl -X POST http://127.0.0.1:8000/predict/risk -H "Content-Type: application/json" -d '{
      "date_time": "2024-01-15T07:00:00", "weather_main": "fog",
      "temp": 265.0, "clouds_all": 90, "rain_1h": 0.0, "snow_1h": 0.0,
      "is_holiday": false
    }'
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field, field_validator

from ml_common import configure_logging
from serving_common import (
    WEATHER_CATEGORIES, FeatureEngineeringError, compute_and_save_scaling_params,
    engineer_feature_row, load_scaling_params,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTERED_NAME = "traffic-risk-classifier"
DEFAULT_ALIAS = "champion"

# Populated at startup by load_model_state(); a module-level dict (rather
# than a global per-variable) so /health and /model-info can report exactly
# what /predict/risk is using, with no risk of the two drifting apart.
STATE: dict = {}


class RawReading(BaseModel):
    """One traffic/weather reading, in the same raw shape as the original
    dataset's own columns -- not pre-engineered features. See
    serving_common.engineer_feature_row for how this becomes a model input."""
    date_time: str = Field(..., description="ISO 8601 datetime, e.g. '2024-01-15T07:00:00'.")
    weather_main: str = Field(..., description=f"One of: {', '.join(WEATHER_CATEGORIES)} (case-insensitive).")
    temp: float = Field(..., description="Air temperature in Kelvin.")
    clouds_all: float = Field(..., ge=0, le=100, description="Cloud cover percentage, 0-100.")
    rain_1h: float = Field(0.0, ge=0, description="Rain in the past hour, mm. Default 0.")
    snow_1h: float = Field(0.0, ge=0, description="Snow in the past hour, mm. Default 0.")
    is_holiday: bool = Field(False, description="Whether this reading falls on a recognized US holiday.")

    @field_validator("weather_main")
    @classmethod
    def _weather_known(cls, v: str) -> str:
        if v.strip().lower() not in WEATHER_CATEGORIES:
            raise ValueError(f"must be one of {WEATHER_CATEGORIES} (case-insensitive), got '{v}'")
        return v


class RiskPrediction(BaseModel):
    high_risk_prediction: int
    high_risk_probability: float
    model_name: str
    model_version: str
    model_alias: str


def load_model_state(tracking_uri: str, registered_name: str, alias: str, scaling_path: Path) -> dict:
    """Loads the aliased model + its registry metadata + the persisted
    feature-scaling constants once, at startup -- not per-request. Raising
    here (rather than swallowing the error) is deliberate: an API that can't
    load its model should fail to start, not silently serve garbage."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(registered_name, alias)
    except MlflowException as e:
        raise RuntimeError(
            f"No model version aliased '{alias}' for registered model '{registered_name}' "
            f"(tracking store: {tracking_uri}). Run model_registry.py (Task 6.1) first."
        ) from e

    model = mlflow.sklearn.load_model(f"models:/{registered_name}@{alias}")
    run = client.get_run(mv.run_id)
    scaling = load_scaling_params(scaling_path)

    logger.info(
        "Loaded '%s'@%s -> registry version %s (run_id=%s, params=%s, metrics=%s)",
        registered_name, alias, mv.version, mv.run_id, run.data.params, run.data.metrics,
    )
    return {
        "model": model, "scaling": scaling, "registered_name": registered_name,
        "alias": alias, "version": mv.version, "run_id": mv.run_id,
        "params": run.data.params, "metrics": run.data.metrics,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_model_state(
        STATE["_startup_args"]["tracking_uri"], STATE["_startup_args"]["registered_name"],
        STATE["_startup_args"]["alias"], STATE["_startup_args"]["scaling_path"],
    ))
    yield
    STATE.clear()


app = FastAPI(
    title="Smart City Traffic Intelligence -- Risk Prediction API",
    description="Deployment mock-up serving the Task 1/6 traffic-risk classifier from MLflow's Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_name": STATE["registered_name"],
        "model_version": STATE["version"],
        "model_alias": STATE["alias"],
    }


@app.get("/model-info")
def model_info():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": STATE["registered_name"],
        "model_version": STATE["version"],
        "model_alias": STATE["alias"],
        "run_id": STATE["run_id"],
        "params": STATE["params"],
        "metrics": STATE["metrics"],
    }


@app.post("/predict/risk", response_model=RiskPrediction)
def predict_risk(reading: RawReading):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = engineer_feature_row(reading.model_dump(), STATE["scaling"])
    except FeatureEngineeringError as e:
        logger.warning("Rejected prediction request: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    model = STATE["model"]
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    logger.info(
        "Prediction served: input=%s -> high_risk=%d (p=%.4f) [model v%s@%s]",
        reading.model_dump(), prediction, probability, STATE["version"], STATE["alias"],
    )
    return RiskPrediction(
        high_risk_prediction=prediction, high_risk_probability=probability,
        model_name=STATE["registered_name"], model_version=str(STATE["version"]), model_alias=STATE["alias"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Serve the traffic-risk classifier over HTTP (FastAPI/uvicorn).")
    parser.add_argument("--tracking-uri", type=str, default=f"sqlite:///{here}/mlflow.db",
                         help="MLflow tracking store URI (default: local SQLite, same store Tasks 4/6.1 use).")
    parser.add_argument("--registered-name", type=str, default=DEFAULT_REGISTERED_NAME,
                         help="Registered model name to serve (default: traffic-risk-classifier).")
    parser.add_argument("--alias", type=str, default=DEFAULT_ALIAS,
                         help="Registry alias to serve (default: champion).")
    parser.add_argument("--input", type=Path, default=here.parent / "pipeline" / "traffic_features.csv",
                         help="Path to Part 2's feature-engineered CSV, used only to (re)compute the "
                              "feature-scaling parameters if they aren't already cached.")
    parser.add_argument("--scaling-path", type=Path, default=here / "artifacts" / "feature_scaling_params.json",
                         help="Where to read/write the persisted feature-scaling parameters.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", type=Path, default=here / "deploy_api.log",
                         help="Path to write the log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Command invoked: deploy_api with arguments: %s", vars(args))

    if not args.scaling_path.exists():
        logger.info("No cached feature-scaling parameters at %s -- computing from %s", args.scaling_path, args.input)
        try:
            compute_and_save_scaling_params(args.input, args.scaling_path)
        except (FileNotFoundError, OSError) as e:
            logger.error("Could not compute feature-scaling parameters: %s", e, exc_info=True)
            return 1

    STATE["_startup_args"] = {
        "tracking_uri": args.tracking_uri, "registered_name": args.registered_name,
        "alias": args.alias, "scaling_path": args.scaling_path,
    }

    try:
        # Fail fast on a bad registry/alias BEFORE handing control to
        # uvicorn, so a misconfigured deployment shows one clean logged
        # error and a non-zero exit rather than an API that starts serving
        # 503s forever.
        load_model_state(args.tracking_uri, args.registered_name, args.alias, args.scaling_path)
    except RuntimeError as e:
        logger.error("Deployment startup aborted: %s", e)
        return 1

    logger.info("Starting API server on %s:%d ...", args.host, args.port)
    # log_config=None is required here: uvicorn.run()'s default logging
    # setup calls logging.config.dictConfig(), which -- regardless of its
    # own disable_existing_loggers=False setting -- unconditionally closes
    # every handler already registered process-wide (a documented Python
    # logging.config quirk, not an app bug). That silently breaks this
    # project's own FileHandler (configure_logging(), above): console
    # logging keeps working by chance, since closing a StreamHandler never
    # closes sys.stdout, but the log FILE stops receiving any request-time
    # log line after the server starts. Caught by testing this script's
    # actual behavior end-to-end, not by reading the code -- see
    # RESULTS_TASK6.md.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
