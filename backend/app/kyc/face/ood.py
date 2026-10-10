import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List

class OODDetector:
    """
    Section 4.6 Uncertainty and Out-of-Distribution (OOD) Detector.
    Identifies non-photographic inputs (drawings, cartoons, sketches, graphics)
    that violate the natural human skin distribution, returning CANNOT_ASSESS.
    """
    def __init__(self, color_entropy_min: float = 2.5, natural_gradient_min: float = 8.0):
        self.color_entropy_min = color_entropy_min
        self.natural_gradient_min = natural_gradient_min

    def assess_distribution(self, face_crop: Image.Image) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Returns: (is_in_distribution, reasons, metrics)
        """
        np_rgb = np.array(face_crop)
        reasons = []

        # 1. Discrete Shannon Entropy of color distribution
        counts, _ = np.histogram(np_rgb, bins=32, range=(0, 256))
        p = counts / float(counts.sum())
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))

        if entropy < self.color_entropy_min:
            reasons.append("NON_PHOTOGRAPHIC_TEXTURE_LOW_ENTROPY")

        # 2. Skin Color Gamut Check in HSV
        hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        skin_mask = ((h <= 28) | (h >= 165)) & (s >= 15) & (s <= 220) & (v >= 35)
        skin_fraction = float(np.mean(skin_mask))

        if skin_fraction < 0.10:
            reasons.append("UNNATURAL_FACIAL_PIGMENTATION_SPECTRUM")

        # 3. Continuous Gradient Dispersion
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_std = float(np.std(grad_mag))

        if grad_std < self.natural_gradient_min:
            reasons.append("ARTIFICIAL_VECTOR_FLAT_GRADIENTS")

        metrics = {
            "color_entropy": round(entropy, 2),
            "skin_fraction": round(skin_fraction, 3),
            "gradient_dispersion": round(grad_std, 2)
        }

        is_in_dist = len(reasons) == 0
        return is_in_dist, reasons, metrics
