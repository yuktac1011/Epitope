import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.app.kyc.inference import KYCDetector

app = FastAPI(
    title="FedShield KYC Screening API v1",
    version="1.0.0",
    description="Production-grade, privacy-preserving face manipulation screening endpoint for digital onboarding."
)

detector = KYCDetector()

class ReasonItem(BaseModel):
    code: str
    weight: float = 1.0

class QualityInfo(BaseModel):
    n_faces: int
    usable: bool
    retake_reasons: List[str] = []
    metrics: Dict[str, Any] = {}

class SignalItem(BaseModel):
    available: bool
    score: Optional[float] = None
    confidence: Optional[float] = None

class KYCScoreResponse(BaseModel):
    request_id: str
    decision: str = Field(..., description="LOW_RISK | UNCERTAIN | HIGH_RISK | RETAKE | CANNOT_ASSESS")
    route: str = Field(..., description="auto_pass | step_up | fraud_desk | retake")
    risk: float = Field(..., ge=0.0, le=1.0, description="Calibrated synthetic risk probability")
    flags: List[str] = []
    reasons: List[Dict[str, Any]] = []
    quality: QualityInfo
    signals: Dict[str, Dict[str, Any]]
    model_version: str
    thresholds_version: str
    latency_ms: int
    input_sha256: Optional[str] = None

@app.post("/kyc/photo/score", response_model=KYCScoreResponse)
async def score_kyc_photo(file: UploadFile = File(...)):
    """
    Section 4.8 API v1 Endpoint.
    Accepts multipart image upload, executes full multi-stage screening pipeline,
    and returns a structured, explainable decision audit record.
    Images are processed strictly in memory and are never stored.
    """
    try:
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        result = detector.assess_photo(raw_bytes, explain=False)

        # Exclude PIL cropped_face from JSON response
        result.pop("cropped_face", None)
        result.pop("explanation_heatmap", None)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image screening failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fedshield-kyc", "version": detector.version}
