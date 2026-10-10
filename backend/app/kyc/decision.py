import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

def route_kyc_decision(
    risk_score: float,
    t_low: float = 0.15,
    t_high: float = 0.75,
    flags: Optional[List[str]] = None,
    quality_usable: bool = True,
    quality_reasons: Optional[List[str]] = None,
    detection_status: str = "OK",
    exposure_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Structured, calibrated decision routing implementing Section 3.6 of KYC Phases 1-2.
    Decisions: RETAKE | CANNOT_ASSESS | LOW_RISK | UNCERTAIN | HIGH_RISK
    Routes: retake | step_up | auto_pass | fraud_desk
    """
    case_id = f"kyc-{uuid.uuid4().hex[:8]}"
    flags = flags or []
    quality_reasons = quality_reasons or []

    # 1. Quality gate failures take precedence -> RETAKE
    if not quality_usable:
        return {
            "case_id": case_id,
            "decision": "RETAKE",
            "route": "retake",
            "risk": risk_score,
            "action": f"Ask user to retake photo: {', '.join(quality_reasons)}",
            "priority": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }

    # 2. Face detection anomalies -> CANNOT_ASSESS (never auto-approve)
    if detection_status != "OK":
        return {
            "case_id": case_id,
            "decision": "CANNOT_ASSESS",
            "route": "step_up",
            "risk": risk_score,
            "action": f"Abstain and route to in-person / step-up verification ({detection_status})",
            "priority": 0.5 * exposure_weight,
            "timestamp": datetime.utcnow().isoformat()
        }

    # 3. High risk threshold -> HIGH_RISK (Manual Fraud Desk)
    if risk_score >= t_high:
        priority = round(risk_score * exposure_weight, 4)
        return {
            "case_id": case_id,
            "decision": "HIGH_RISK",
            "route": "fraud_desk",
            "risk": risk_score,
            "action": "Manual review / Enhanced fraud desk verification",
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }

    # 4. Non-live capture flag OR risk in ambiguous zone -> UNCERTAIN (Step-up Video Liveness)
    if risk_score >= t_low or "NON_LIVE_CAPTURE_SUSPECTED" in flags:
        priority = round(risk_score * exposure_weight, 4)
        reason_note = "Non-live capture flag" if "NON_LIVE_CAPTURE_SUSPECTED" in flags else "Ambiguous score"
        return {
            "case_id": case_id,
            "decision": "UNCERTAIN",
            "route": "step_up",
            "risk": risk_score,
            "action": f"Route to active video liveness challenge ({reason_note})",
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }

    # 5. Clean, low-risk capture -> LOW_RISK (Auto-pass)
    return {
        "case_id": case_id,
        "decision": "LOW_RISK",
        "route": "auto_pass",
        "risk": risk_score,
        "action": "Continue normal KYC onboarding workflow",
        "priority": 0.0,
        "timestamp": datetime.utcnow().isoformat()
    }

# Backward compatibility alias
route_kyc_case = route_kyc_decision
