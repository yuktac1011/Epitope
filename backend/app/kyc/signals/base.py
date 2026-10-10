from dataclasses import dataclass, field
from typing import Protocol, Optional, List, Dict, Any

@dataclass
class SignalResult:
    name: str
    available: bool
    score: Optional[float]
    confidence: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Sample:
    frames: List[Any]
    timestamps_ms: List[int]
    audio: Optional[bytes] = None
    capture_meta: Dict[str, Any] = field(default_factory=dict)
    challenge: Optional[Dict[str, Any]] = None

class Signal(Protocol):
    name: str
    def score(self, sample: Sample) -> SignalResult:
        ...
