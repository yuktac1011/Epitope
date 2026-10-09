"""
Pydantic request/response schemas for the Laya API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Generic predict request: caller provides the raw state and questions
    exactly as Laya expects them.
    """
    state: Any = Field(
        ...,
        description="The text/data to classify. Can be a string, dict, or list.",
        examples=[
            {"subject": "Charged twice", "body": "Refund the duplicate today or we cancel."},
            "Customer says they were charged twice for order #1234.",
        ],
    )
    questions: Dict[str, dict] = Field(
        ...,
        description="Dict of question definitions. Each key is a question id, "
                    "each value has 'type' (choice|score|noul), 'instructions', "
                    "and optionally 'criteria'.",
    )
    lang: Optional[str] = Field(
        None,
        description="ISO language code. Pass this when you know the language "
                    "to avoid mis-routing (see SKILL.md).",
    )


class TriageRequest(BaseModel):
    """
    Convenience triage endpoint: caller provides transaction context,
    and we use the pre-built fraud_triage_questions.
    """
    text: str = Field(
        ...,
        description="Free-text description of the transaction, alert, or customer message.",
        examples=["Customer reports unauthorized withdrawal of $5,000 from their account."],
    )
    transaction_type: Optional[str] = Field(
        None, description="Type of transaction (e.g. TRANSFER, CASH_OUT)."
    )
    amount: Optional[float] = Field(
        None, description="Transaction amount.", ge=0
    )
    sender: Optional[str] = Field(None, description="Sender account name/ID.")
    receiver: Optional[str] = Field(None, description="Receiver account name/ID.")
    balance_before: Optional[float] = Field(
        None, description="Account balance before the transaction.", ge=0
    )
    balance_after: Optional[float] = Field(
        None, description="Account balance after the transaction.", ge=0
    )
    question_set: Optional[str] = Field(
        "triage",
        description="Which question set to use: 'triage', 'classification', or 'routing'.",
    )
    confidence_threshold: Optional[float] = Field(
        0.85,
        description="Minimum probability to act on the top prediction. "
                    "Below this, the result is flagged for review.",
        ge=0.0,
        le=1.0,
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnswerDetail(BaseModel):
    """Single question answer from Laya."""
    question_id: str
    question_type: str  # "choice" | "score" | "noul"
    result: Any  # the winning choice, score float, or noul probability
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    above_threshold: bool = True


class PredictResponse(BaseModel):
    """Response from the generic /predict endpoint."""
    answers: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = Field(
        None, description="Full raw Laya response for debugging."
    )


class TriageResponse(BaseModel):
    """Response from the /triage convenience endpoint."""
    fraud_category: AnswerDetail
    urgency: AnswerDetail
    churn_risk: AnswerDetail
    needs_review: bool = Field(
        False,
        description="True if any answer fell below the confidence threshold.",
    )
    review_reasons: List[str] = Field(
        default_factory=list,
        description="Which questions triggered the review flag and why.",
    )
    usage: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str  # "ready" | "loading" | "unavailable"
    model_loaded: bool
    checkpoint: Optional[str] = None
    device: Optional[str] = None
