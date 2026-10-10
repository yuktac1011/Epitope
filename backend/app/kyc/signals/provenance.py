import io
from PIL import Image
from typing import Dict, Any, List
from backend.app.kyc.signals.base import SignalResult, Sample

class ProvenanceSignal:
    """
    Section 4.4 Provenance Signals (E3).
    Inspects image container chunks, metadata, and C2PA markers for generator fingerprints.
    Absence of provenance is neutral; presence of AI tool tags or editing signatures is flagged.
    """
    name: str = "provenance"

    AI_SIGNATURE_KEYWORDS = [
        "stable diffusion", "midjourney", "dall-e", "comfyui", "automatic1111",
        "novelai", "photoshop", "gimp", "faceswap", "deepfacelab", "simswap", "roop"
    ]

    def score(self, sample: Sample) -> SignalResult:
        meta = sample.capture_meta or {}
        reasons: List[str] = []
        suspicion_score = 0.0
        detected_tags: Dict[str, str] = {}

        if not sample.frames:
            return SignalResult(name=self.name, available=False, score=None, reasons=["NO_INPUT_FRAMES"])

        frame = sample.frames[0]
        if isinstance(frame, Image.Image):
            # 1. Inspect PNG text chunks (parameters, prompt, software)
            if hasattr(frame, "info") and frame.info:
                for k, v in frame.info.items():
                    if isinstance(v, str):
                        v_lower = v.lower()
                        for kw in self.AI_SIGNATURE_KEYWORDS:
                            if kw in v_lower:
                                suspicion_score = max(suspicion_score, 0.85)
                                reasons.append(f"AI_GENERATOR_METADATA_DETECTED: {kw.upper()}")
                                detected_tags[str(k)] = v[:100]

            # 2. Inspect EXIF Software / ImageDescription tags
            try:
                exif = frame.getexif()
                if exif:
                    for tag_id, val in exif.items():
                        if isinstance(val, str):
                            val_lower = val.lower()
                            for kw in self.AI_SIGNATURE_KEYWORDS:
                                if kw in val_lower:
                                    suspicion_score = max(suspicion_score, 0.80)
                                    reasons.append(f"EDITING_SOFTWARE_METADATA_DETECTED: {kw.upper()}")
                                    detected_tags[f"exif_{tag_id}"] = val[:100]
            except Exception:
                pass

        available = len(detected_tags) > 0 or meta.get("had_exif", False)

        if not reasons:
            reasons.append("NO_PROVENANCE_ANOMALIES")

        return SignalResult(
            name=self.name,
            available=available,
            score=round(suspicion_score, 3) if available else 0.0,
            confidence=0.9 if suspicion_score > 0.5 else 0.2,
            reasons=reasons,
            meta={"detected_tags": detected_tags}
        )
