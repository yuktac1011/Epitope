import os
import io
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from PIL import Image

from backend.app.kyc.io.loader import safe_load_image, InputValidationError
from backend.app.kyc.face.detect import FaceDetector, FaceBox
from backend.app.kyc.face.crop import canonical_face_crop
from backend.app.kyc.face.quality import QualityGate
from backend.app.kyc.decision import route_kyc_decision
from backend.app.kyc.inference import KYCDetector

def create_synthetic_image(width=300, height=300, color=(128, 128, 128), fmt="PNG"):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue(), img

def test_loader_rejects_empty_file():
    with pytest.raises(InputValidationError, match="Empty input file"):
        safe_load_image(b"")

def test_loader_rejects_unsupported_format():
    buf = io.BytesIO()
    img = Image.new("RGB", (50, 50))
    img.save(buf, format="GIF")
    with pytest.raises(InputValidationError, match="Unsupported format"):
        safe_load_image(buf.getvalue())

def test_loader_safe_decode_and_alpha():
    rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    loaded, meta = safe_load_image(buf.getvalue())
    assert loaded.mode == "RGB"
    assert meta["had_alpha"] is True
    assert meta["width"] == 100

def test_quality_gate_detects_blur_and_darkness():
    gate = QualityGate(min_blur_variance=45.0, min_mean_luminance=35.0)
    # 1. Perfectly flat gray image -> variance of Laplacian is 0 -> TOO_BLURRY
    flat_crop = Image.new("RGB", (100, 100), (120, 120, 120))
    box = FaceBox(x=10, y=10, w=80, h=80, score=0.99)
    res_blur = gate.assess(flat_crop, box)
    assert not res_blur.is_usable
    assert "TOO_BLURRY" in res_blur.reasons

    # 2. Very dark image -> mean luminance < 35 -> TOO_DARK
    dark_crop = Image.new("RGB", (100, 100), (10, 10, 10))
    res_dark = gate.assess(dark_crop, box)
    assert not res_dark.is_usable
    assert "TOO_DARK" in res_dark.reasons

def test_decision_routing_table():
    # Retake priority
    d_retake = route_kyc_decision(risk_score=0.1, quality_usable=False, quality_reasons=["TOO_BLURRY"])
    assert d_retake["decision"] == "RETAKE"
    assert d_retake["route"] == "retake"

    # Cannot assess
    d_no_face = route_kyc_decision(risk_score=0.1, detection_status="NO_FACE")
    assert d_no_face["decision"] == "CANNOT_ASSESS"
    assert d_no_face["route"] == "step_up"

    # Clean low risk
    d_low = route_kyc_decision(risk_score=0.08, t_low=0.15)
    assert d_low["decision"] == "LOW_RISK"
    assert d_low["route"] == "auto_pass"

    # Ambiguous / uncertain
    d_unc = route_kyc_decision(risk_score=0.35, t_low=0.15, t_high=0.75)
    assert d_unc["decision"] == "UNCERTAIN"
    assert d_unc["route"] == "step_up"

    # High risk
    d_high = route_kyc_decision(risk_score=0.88, t_high=0.75)
    assert d_high["decision"] == "HIGH_RISK"
    assert d_high["route"] == "fraud_desk"

def test_full_pipeline_on_dataset_images():
    detector = KYCDetector()
    if os.path.exists("test_real_photo.png"):
        res_real = detector.assess_photo("test_real_photo.png")
        assert "decision" in res_real
        assert "risk" in res_real
        assert "quality" in res_real
        assert res_real["quality"]["n_faces"] >= 1
        print("Real image pipeline result:", res_real["decision"], "Risk:", res_real["risk"])

    if os.path.exists("test_fake_photo.png"):
        res_fake = detector.assess_photo("test_fake_photo.png", explain=True)
        assert "decision" in res_fake
        assert res_fake["explanation_heatmap"] is not None
        assert "signals" in res_fake
        assert "frequency" in res_fake["signals"]
        print("Fake image pipeline result:", res_fake["decision"], "Risk:", res_fake["risk"])

def test_ood_detector_identifies_unnatural_art():
    from backend.app.kyc.face.ood import OODDetector
    ood = OODDetector()
    # Create artificial high-saturation flat graphics (non-skin)
    flat_art = Image.new("RGB", (224, 224), (0, 255, 0))
    is_in_dist, reasons, _ = ood.assess_distribution(flat_art)
    assert not is_in_dist
    assert len(reasons) > 0

def test_fastapi_v1_endpoint():
    from fastapi.testclient import TestClient
    from backend.app.kyc.api import app
    client = TestClient(app)
    
    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    
    # Photo score check
    if os.path.exists("test_real_photo.png"):
        with open("test_real_photo.png", "rb") as f:
            files = {"file": ("test_real.png", f, "image/png")}
            resp = client.post("/kyc/photo/score", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert "decision" in data
            assert "risk" in data
            assert "signals" in data
