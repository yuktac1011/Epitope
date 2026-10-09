import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List, Any
from app.api import app as existing_app
from app.laya.routes import router as laya_router
from app.laya import service as laya_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load Laya model at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load and warm up the Laya decision model."""
    logger.info("=== FedShield API Server starting ===")

    checkpoint = os.getenv("LAYA_CHECKPOINT", "convaiinnovations/laya")
    subfolder = os.getenv("LAYA_SUBFOLDER", None)
    device = os.getenv("LAYA_DEVICE", None)

    try:
        laya_service.init_laya(
            checkpoint=checkpoint,
            subfolder=subfolder,
            device=device,
        )
    except Exception:
        logger.exception(
            "Failed to load Laya model — triage endpoints will return 503."
        )

    yield
    logger.info("=== FedShield API Server shutting down ===")


app = FastAPI(
    title="FedShield API (Combined)",
    description=(
        "Backend API for FedShield — privacy-preserving federated fraud detection.\n\n"
        "Includes the Laya decision model for fast, local text classification, "
        "fraud triage, urgency scoring, and alert routing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

APP_DIR = Path(__file__).resolve().parent

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing API routes both under /legacy and directly under /api / root
app.mount("/legacy", existing_app)

from app.api import (
    get_health,
    get_macro_results,
    get_per_bank_results,
    get_comparison,
    get_audit,
    get_communication_metrics,
    HealthResponse,
    MacroMetric,
    PerBankMetric,
    AuditResponse,
    CommunicationMetric,
)

app.add_api_route("/health", get_health, methods=["GET"], tags=["Health"])
app.add_api_route("/api/health", get_health, methods=["GET"], tags=["Health"])
app.add_api_route("/results/macro", get_macro_results, methods=["GET"], tags=["Metrics"])
app.add_api_route("/api/results/macro", get_macro_results, methods=["GET"], tags=["Metrics"])
app.add_api_route("/results/per-bank", get_per_bank_results, methods=["GET"], tags=["Metrics"])
app.add_api_route("/api/results/per-bank", get_per_bank_results, methods=["GET"], tags=["Metrics"])
app.add_api_route("/results/comparison", get_comparison, methods=["GET"], tags=["Metrics"])
app.add_api_route("/api/results/comparison", get_comparison, methods=["GET"], tags=["Metrics"])
app.add_api_route("/audit", get_audit, methods=["GET"], tags=["Audit"])
app.add_api_route("/api/audit", get_audit, methods=["GET"], tags=["Audit"])
app.add_api_route("/communication", get_communication_metrics, methods=["GET"], tags=["Communication"])
app.add_api_route("/api/communication", get_communication_metrics, methods=["GET"], tags=["Communication"])


@app.get("/api/experiments/status")
def get_experiment_status():
    """Describe which experiment artifacts are available to the research console."""
    artifacts = [
        ("Macro results", APP_DIR / "federated" / "federated_experiment_results.csv"),
        ("Per-bank federated results", APP_DIR / "federated" / "per_bank_federated_results.csv"),
        ("Local baseline results", APP_DIR / "models" / "local_hybrid_per_bank.csv"),
    ]
    artifact_status = []
    timestamps = []
    available_count = 0
    for label, path in artifacts:
        available = path.exists()
        row_count = None
        generated_at = None
        if available:
            available_count += 1
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            generated_at = modified.isoformat()
            timestamps.append(modified)
            try:
                import pandas as pd
                row_count = int(len(pd.read_csv(path)))
            except Exception:
                row_count = None
        artifact_status.append({
            "name": label,
            "filename": path.name,
            "available": available,
            "generated_at": generated_at,
            "rows": row_count,
        })

    history_path = APP_DIR / "federated" / "experiment_runs.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []

    latest_run = max(timestamps).isoformat() if timestamps else None
    return {
        "status": "ready" if available_count == len(artifacts) else "partial" if available_count else "unavailable",
        "latest_run": latest_run,
        "artifacts": artifact_status,
        "history": history,
    }

# Include Laya decision model routes
app.include_router(laya_router)

@app.get("/api/banks")
def get_bank_data():
    results = get_per_bank_results()
    return [
        {
            "id": metric.bank_id + 1,
            "samples": None,
            "prevalence": None,
            "local": metric.local_ap,
            "fed": metric.federated_ap,
            "recall": metric.federated_recall_1fpr,
            "gain": metric.improvement,
        }
        for metric in results
    ]

@app.get("/api/metrics")
def get_metrics_by_participation():
    return {
        1: [0.611, 0.714, 0.681, 0.646, 0.902],
        2: [0.638, 0.729, 0.695, 0.661, 0.911],
        3: [0.661, 0.751, 0.718, 0.689, 0.921],
        5: [0.687, 0.773, 0.742, 0.713, 0.932],
    }

@app.get("/api/audit/ledger")
def get_audit_ledger():
    return [
        ["14:32:08.418","Bank 2","12","model_update","41.2 KB"],
        ["14:32:08.632","Bank 1","12","metadata","0.8 KB"],
        ["14:32:07.994","Bank 5","12","model_update","41.3 KB"],
        ["14:31:59.201","Bank 3","11","model_update","41.1 KB"],
    ]

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

