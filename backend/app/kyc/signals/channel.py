import numpy as np
from PIL import Image
from typing import Optional
from backend.app.kyc.signals.base import SignalResult, Sample

class ChannelSignal:
    """
    Evaluates capture-channel signals to distinguish live captures
    from screenshots or screen re-captures without falsely labeling them as deepfakes.
    """
    name: str = "channel"

    # Common smartphone aspect ratios (height / width in portrait)
    PHONE_ASPECT_RATIOS = [
        16.0 / 9.0,   # 1.777 (Older iPhones, many Androids)
        19.5 / 9.0,   # 2.166 (iPhone X-15, modern Samsung)
        20.0 / 9.0,   # 2.222 (Modern Androids)
        19.0 / 9.0,   # 2.111
        21.0 / 9.0    # 2.333 (Sony Xperia)
    ]

    def score(self, sample: Sample) -> SignalResult:
        meta = sample.capture_meta or {}
        reasons = []
        indicators = 0.0

        w = meta.get("width")
        h = meta.get("height")
        had_exif = meta.get("had_exif", False)
        img_format = meta.get("format", "").upper()

        if w and h:
            aspect = max(w, h) / float(min(w, h))
            # 1. Aspect ratio match to phone screens
            matches_screen_aspect = any(abs(aspect - target) < 0.02 for target in self.PHONE_ASPECT_RATIOS)
            if matches_screen_aspect:
                indicators += 0.35
                reasons.append("SCREEN_ASPECT_RATIO_MATCH")

        # 2. Missing EXIF metadata on PNG upload (common screenshot artifact)
        if img_format == "PNG" and not had_exif:
            indicators += 0.25
            reasons.append("PNG_WITHOUT_CAMERA_METADATA")

        # 3. Letterbox / uniform border detection
        if sample.frames:
            frame = sample.frames[0]
            if isinstance(frame, Image.Image):
                np_img = np.array(frame)
                # Check top and bottom 2% rows for zero variance (solid black/white bars)
                if np_img.ndim >= 2:
                    h_px = np_img.shape[0]
                    bar_h = max(1, int(h_px * 0.02))
                    top_bar = np_img[:bar_h, :]
                    bottom_bar = np_img[-bar_h:, :]

                    if np.std(top_bar) < 2.0 or np.std(bottom_bar) < 2.0:
                        indicators += 0.30
                        reasons.append("UNIFORM_LETTERBOX_BARS_DETECTED")

        channel_score = min(1.0, indicators)
        if channel_score >= 0.50:
            reasons.append("NON_LIVE_CAPTURE_SUSPECTED")

        return SignalResult(
            name=self.name,
            available=True,
            score=round(channel_score, 3),
            reasons=reasons,
            meta={"indicators": indicators}
        )
