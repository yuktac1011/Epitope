from typing import Dict, List, Any
from dataclasses import dataclass, field
from backend.app.kyc.signals.base import SignalResult

@dataclass
class FusionResult:
    calibrated_risk: float
    flags: List[str] = field(default_factory=list)
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    signal_scores: Dict[str, Any] = field(default_factory=dict)

class RiskFusion:
    """
    Section 4.5 Fusion Engine.
    Missing-signal aware weighted logistic fusion across:
    - E1: Artifact Spatial CNN (0.65)
    - E2: Capture Channel / Screenshot (0.15)
    - Frequency FFT Checkerboard (0.10)
    - E3: Provenance Metadata (0.10)
    """
    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        self.weights = weights or {
            "artifact": 0.65,
            "channel": 0.15,
            "frequency": 0.10,
            "provenance": 0.10
        }

    def fuse(self, signal_results: Dict[str, SignalResult]) -> FusionResult:
        flags = []
        reasons = []
        scores_summary = {}

        total_weight = 0.0
        weighted_sum = 0.0

        for name, sig_res in signal_results.items():
            w = self.weights.get(name, 0.10)
            if sig_res and sig_res.available and sig_res.score is not None:
                scores_summary[name] = {
                    "score": round(sig_res.score, 4),
                    "available": True,
                    "confidence": sig_res.confidence
                }
                weighted_sum += sig_res.score * w
                total_weight += w

                # Flags
                if name == "channel" and "NON_LIVE_CAPTURE_SUSPECTED" in sig_res.reasons:
                    flags.append("NON_LIVE_CAPTURE_SUSPECTED")
                if name == "provenance" and any("AI_GENERATOR" in r for r in sig_res.reasons):
                    flags.append("AI_GENERATOR_METADATA_FLAG")

                # Reason codes
                for r in sig_res.reasons:
                    reasons.append({"code": r, "signal": name, "weight": round(w, 2)})
            else:
                scores_summary[name] = {"score": None, "available": False}

        calibrated_risk = (weighted_sum / total_weight) if total_weight > 0 else 0.50

        # Non-live capture flag raises floor risk to Uncertain
        if "NON_LIVE_CAPTURE_SUSPECTED" in flags and calibrated_risk < 0.25:
            calibrated_risk = 0.30

        return FusionResult(
            calibrated_risk=round(calibrated_risk, 4),
            flags=flags,
            reasons=reasons,
            signal_scores=scores_summary
        )
