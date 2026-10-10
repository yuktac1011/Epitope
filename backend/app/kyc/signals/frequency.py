import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List
from backend.app.kyc.signals.base import SignalResult, Sample

class FrequencySignal:
    """
    Section 4.2 Frequency-Domain Signal.
    Analyzes the 2D Fourier transform power spectrum of the face crop
    to detect periodic high-frequency checkerboard artifacts characteristic
    of convolutional upsampling in GAN and Diffusion face synthesizers.
    """
    name: str = "frequency"

    def score(self, sample: Sample) -> SignalResult:
        if not sample.frames:
            return SignalResult(name=self.name, available=False, score=None, reasons=["NO_INPUT_FRAMES"])

        frame = sample.frames[0]
        if isinstance(frame, Image.Image):
            gray = np.array(frame.convert("L"), dtype=np.float32)
        else:
            gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Compute 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2

        # 2. Extract High-Frequency Annulus
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_radius = min(cx, cy)

        # High frequency band: 0.5 to 0.9 of max radius
        high_freq_mask = (dist_from_center >= 0.5 * max_radius) & (dist_from_center <= 0.9 * max_radius)
        low_freq_mask = (dist_from_center <= 0.25 * max_radius)

        high_freq_power = np.mean(magnitude_spectrum[high_freq_mask])
        low_freq_power = np.mean(magnitude_spectrum[low_freq_mask])

        # 3. Peak-to-average ratio in high frequency (detects periodic spikes)
        hf_vals = magnitude_spectrum[high_freq_mask]
        hf_peak_ratio = (np.max(hf_vals) / (np.mean(hf_vals) + 1e-6)) if len(hf_vals) > 0 else 1.0

        # High frequency ratio compared to natural decay
        energy_ratio = high_freq_power / (low_freq_power + 1e-6)

        reasons: List[str] = []
        score = 0.0

        # Synthetic generator checkerboard artifacts produce high peak ratios
        if hf_peak_ratio > 1.85:
            score += 0.45
            reasons.append("HIGH_FREQUENCY_PERIODIC_PEAKS")

        if energy_ratio > 0.65:
            score += 0.35
            reasons.append("UNNATURAL_HIGH_FREQUENCY_ENERGY")

        score = min(1.0, score)

        return SignalResult(
            name=self.name,
            available=True,
            score=round(score, 3),
            confidence=0.8,
            reasons=reasons,
            meta={
                "hf_peak_ratio": round(float(hf_peak_ratio), 3),
                "energy_ratio": round(float(energy_ratio), 3)
            }
        )
