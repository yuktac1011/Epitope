import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from dataclasses import dataclass, field
from backend.app.kyc.face.detect import FaceBox

@dataclass
class QualityResult:
    is_usable: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class QualityGate:
    def __init__(
        self,
        min_face_size_px: int = 60,
        min_blur_variance: float = 45.0,
        min_mean_luminance: float = 35.0,
        max_mean_luminance: float = 235.0,
        max_clipped_fraction: float = 0.25
    ):
        self.min_face_size_px = min_face_size_px
        self.min_blur_variance = min_blur_variance
        self.min_mean_luminance = min_mean_luminance
        self.max_mean_luminance = max_mean_luminance
        self.max_clipped_fraction = max_clipped_fraction

    def assess(self, face_crop: Image.Image, face_box: FaceBox) -> QualityResult:
        reasons = []

        face_short_side = min(face_box.w, face_box.h)
        if face_short_side < self.min_face_size_px:
            reasons.append("FACE_TOO_SMALL")

        np_crop = np.array(face_crop.convert("L"))

        laplacian = cv2.Laplacian(np_crop, cv2.CV_64F)
        blur_variance = float(laplacian.var())
        if blur_variance < self.min_blur_variance:
            reasons.append("TOO_BLURRY")

        mean_luminance = float(np.mean(np_crop))
        if mean_luminance < self.min_mean_luminance:
            reasons.append("TOO_DARK")
        elif mean_luminance > self.max_mean_luminance:
            reasons.append("OVEREXPOSED")

        clipped_dark = np.sum(np_crop <= 5)
        clipped_bright = np.sum(np_crop >= 250)
        clipped_fraction = float((clipped_dark + clipped_bright) / np_crop.size)
        if clipped_fraction > self.max_clipped_fraction and "OVEREXPOSED" not in reasons and "TOO_DARK" not in reasons:
            reasons.append("BAD_LIGHTING")

        metrics = {
            "face_short_side_px": int(face_short_side),
            "blur_variance": round(blur_variance, 2),
            "mean_luminance": round(mean_luminance, 2),
            "clipped_fraction": round(clipped_fraction, 4)
        }

        is_usable = (len(reasons) == 0)
        return QualityResult(is_usable=is_usable, reasons=reasons, metrics=metrics)
