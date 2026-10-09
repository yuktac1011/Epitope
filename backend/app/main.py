import os
import logging
from contextlib import asynccontextmanager

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

# Include existing API routes
app.mount("/legacy", existing_app)

# Include Laya decision model routes
app.include_router(laya_router)

@app.get("/api/banks")
def get_bank_data():
    return [
        { "id": 1, "samples": "228,450", "prevalence": 0.14, "local": 0.611, "fed": 0.687, "recall": 0.714, "gain": 0.076 },
        { "id": 2, "samples": "194,820", "prevalence": 0.11, "local": 0.574, "fed": 0.662, "recall": 0.681, "gain": 0.088 },
        { "id": 3, "samples": "72,190", "prevalence": 0.06, "local": 0.412, "fed": 0.548, "recall": 0.603, "gain": 0.136 },
        { "id": 4, "samples": "116,340", "prevalence": 0.09, "local": 0.521, "fed": 0.585, "recall": 0.642, "gain": 0.064 },
        { "id": 5, "samples": "301,760", "prevalence": 0.22, "local": 0.704, "fed": 0.691, "recall": 0.733, "gain": -0.013 },
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

