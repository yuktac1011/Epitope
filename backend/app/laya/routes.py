"""
Laya API Routes — FastAPI Router

Endpoints:
  POST /api/laya/predict  — generic state + questions predict
  POST /api/laya/triage   — convenience fraud triage
  GET  /api/laya/health   — model health check
"""

import logging
from fastapi import APIRouter, HTTPException

from app.laya import service
from app.laya.schemas import (
    PredictRequest,
    PredictResponse,
    TriageRequest,
    TriageResponse,
    HealthResponse,
    AnswerDetail,
)
from app.laya.questions import (
    fraud_triage_questions,
    transaction_classification_questions,
    alert_routing_questions,
    build_transaction_state,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/laya", tags=["Laya Decision Model"])


# ---------------------------------------------------------------------------
# POST /api/laya/predict — generic
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """
    Generic Laya predict endpoint.

    Pass any state (string, dict, or list) and a dict of questions.
    Returns the full Laya result including answers and usage.
    """
    if not service.is_ready():
        raise HTTPException(status_code=503, detail="Laya model is still loading.")

    try:
        result = service.predict(body.state, body.questions)
    except Exception as e:
        logger.error("Laya predict failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        answers=result.get("answers", {}),
        usage=result.get("usage"),
        raw=result,
    )


# ---------------------------------------------------------------------------
# POST /api/laya/triage — convenience fraud triage
# ---------------------------------------------------------------------------
@router.post("/triage", response_model=TriageResponse)
def triage(body: TriageRequest):
    """
    Convenience endpoint for fraud triage.

    Accepts transaction context and returns fraud category, urgency,
    and churn risk with confidence-based review flagging.
    """
    if not service.is_ready():
        raise HTTPException(status_code=503, detail="Laya model is still loading.")

    # --- Build the state from structured fields ---
    state = build_transaction_state(
        transaction_type=body.transaction_type or "",
        amount=body.amount or 0.0,
        sender=body.sender or "",
        receiver=body.receiver or "",
        description=body.text,
        balance_before=body.balance_before or 0.0,
        balance_after=body.balance_after or 0.0,
    )

    # --- Select question set ---
    question_set_map = {
        "triage": fraud_triage_questions,
        "classification": transaction_classification_questions,
        "routing": alert_routing_questions,
    }
    questions_fn = question_set_map.get(body.question_set or "triage", fraud_triage_questions)
    questions = questions_fn()

    try:
        result = service.predict(state, questions)
    except Exception as e:
        logger.error("Laya triage failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Triage prediction failed: {str(e)}")

    answers = result.get("answers", {})
    threshold = body.confidence_threshold

    # --- Build structured response with threshold checks ---
    needs_review = False
    review_reasons = []

    def _make_answer(qid: str, qtype: str) -> AnswerDetail:
        nonlocal needs_review
        raw_answer = answers.get(qid, {})

        if qtype == "choice":
            result_val = raw_answer.get("choice")
            probs = raw_answer.get("probabilities", {})
            conf = raw_answer.get("confidence")
            top_prob = probs.get(result_val, 0.0) if result_val else 0.0
            above = top_prob >= threshold
        elif qtype == "score":
            result_val = raw_answer.get("score")
            probs = raw_answer.get("probabilities", {})
            conf = raw_answer.get("confidence")
            above = (conf or 0.0) >= threshold
        elif qtype == "noul":
            result_val = raw_answer.get("noul")
            probs = None
            conf = raw_answer.get("confidence")
            above = (conf or 0.0) >= threshold
        else:
            result_val = raw_answer
            probs = None
            conf = None
            above = True

        if not above:
            needs_review = True
            review_reasons.append(
                f"{qid}: confidence {conf:.2f} is below threshold {threshold:.2f}"
                if conf is not None
                else f"{qid}: below threshold"
            )

        return AnswerDetail(
            question_id=qid,
            question_type=qtype,
            result=result_val,
            confidence=conf,
            probabilities=probs,
            above_threshold=above,
        )

    # Build response depending on which question set was used
    if body.question_set == "classification":
        fraud_cat = _make_answer("transaction_type", "choice")
        urg = _make_answer("risk_level", "score")
        churn = AnswerDetail(
            question_id="churn_risk",
            question_type="noul",
            result=None,
            confidence=None,
            above_threshold=True,
        )
    elif body.question_set == "routing":
        fraud_cat = _make_answer("department", "choice")
        urg = AnswerDetail(
            question_id="urgency",
            question_type="score",
            result=None,
            confidence=None,
            above_threshold=True,
        )
        churn = _make_answer("needs_escalation", "noul")
    else:
        # Default: triage
        fraud_cat = _make_answer("fraud_category", "choice")
        urg = _make_answer("urgency", "score")
        churn = _make_answer("churn_risk", "noul")

    return TriageResponse(
        fraud_category=fraud_cat,
        urgency=urg,
        churn_risk=churn,
        needs_review=needs_review,
        review_reasons=review_reasons,
        usage=result.get("usage"),
    )


# ---------------------------------------------------------------------------
# GET /api/laya/health — health check
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check: reports whether Laya is loaded and ready.
    """
    ready = service.is_ready()
    agent = service.get_agent()

    device_str = None
    if agent is not None:
        try:
            device_str = str(getattr(agent, "device", "unknown"))
        except Exception:
            device_str = "unknown"

    return HealthResponse(
        status="ready" if ready else "unavailable",
        model_loaded=ready,
        checkpoint="convaiinnovations/laya" if ready else None,
        device=device_str,
    )
