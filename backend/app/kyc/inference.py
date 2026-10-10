import time
import uuid
import os
import yaml
from typing import Union, Dict, Any, Optional
from PIL import Image

from backend.app.kyc.io.loader import safe_load_image
from backend.app.kyc.face.detect import FaceDetector
from backend.app.kyc.face.crop import canonical_face_crop
from backend.app.kyc.face.quality import QualityGate
from backend.app.kyc.signals.base import Sample
from backend.app.kyc.signals.artifact import ArtifactSignal
from backend.app.kyc.signals.channel import ChannelSignal
from backend.app.kyc.fusion import RiskFusion
from backend.app.kyc.decision import route_kyc_decision
from backend.app.kyc.explain import GradCAMExplainer
from backend.app.kyc.signals.frequency import FrequencySignal
from backend.app.kyc.signals.provenance import ProvenanceSignal
from backend.app.kyc.face.ood import OODDetector

class KYCDetector:
    """
    Main orchestrator for the Phase 1 & 2 KYC Photo Screening Pipeline.
    Assembles Safe Loader -> Face Detector -> Quality Gate -> Canonical Crop 
    -> Signals (Artifact, Channel) -> Risk Fusion -> Calibrated Decision.
    """
    def __init__(self, config_path: str = "backend/app/kyc/config/kyc.yaml", model_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        face_cfg = self.config.get("face", {})
        quality_cfg = self.config.get("quality", {})
        decision_cfg = self.config.get("decision", {})

        # Model override if passed directly
        artifact_model_path = model_path or "backend/app/kyc/checkpoints/best_kyc_model.pth"

        self.face_detector = FaceDetector(
            model_path=face_cfg.get("model_path", "backend/app/kyc/models/face_detection_yunet_2023mar.onnx"),
            score_threshold=face_cfg.get("score_threshold", 0.60),
            nms_threshold=face_cfg.get("nms_threshold", 0.30),
            dominant_area_ratio=face_cfg.get("dominant_area_ratio", 0.60)
        )

        self.quality_gate = QualityGate(
            min_face_size_px=quality_cfg.get("min_face_size_px", 60),
            min_blur_variance=quality_cfg.get("min_blur_variance", 45.0),
            min_mean_luminance=quality_cfg.get("min_mean_luminance", 35.0),
            max_mean_luminance=quality_cfg.get("max_mean_luminance", 235.0),
            max_clipped_fraction=quality_cfg.get("max_clipped_fraction", 0.25)
        )

        self.artifact_signal = ArtifactSignal(checkpoint_path=artifact_model_path)
        self.channel_signal = ChannelSignal()
        self.frequency_signal = FrequencySignal()
        self.provenance_signal = ProvenanceSignal()
        self.ood_detector = OODDetector()
        self.fusion = RiskFusion()
        self.explainer = GradCAMExplainer(self.artifact_signal.model)

        self.t_low = decision_cfg.get("t_low", 0.15)
        self.t_high = decision_cfg.get("t_high", 0.75)
        self.crop_margin_ratio = face_cfg.get("crop_margin_ratio", 1.25)
        self.target_size = tuple(face_cfg.get("target_size", [224, 224]))

        self.version = self.config.get("version", "kyc-photo-1.0.0")
        self.thresholds_version = self.config.get("thresholds_version", "2026-10-10")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def assess_photo(self, input_data: Union[str, bytes, Image.Image], explain: bool = False) -> Dict[str, Any]:
        """
        Executes full Phase 1 screening pipeline.
        Returns complete API v1 structured response matching Section 4.8.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # [A] Safe Decode & Normalize
        if isinstance(input_data, Image.Image):
            pil_img = input_data.convert("RGB")
            meta = {
                "sha256": "in-memory",
                "format": "PIL",
                "width": pil_img.size[0],
                "height": pil_img.size[1],
                "had_exif": False,
                "had_alpha": False,
                "animated": False
            }
        else:
            pil_img, meta = safe_load_image(input_data)

        # [B] Face Detection
        detect_result = self.face_detector.detect(pil_img)

        # Handle NO_FACE or MULTIPLE_FACES
        if detect_result.status != "OK" or not detect_result.primary_face:
            decision_record = route_kyc_decision(
                risk_score=0.50,
                detection_status=detect_result.status
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "request_id": request_id,
                "decision": decision_record["decision"],
                "route": decision_record["route"],
                "risk": 0.50,
                "action": decision_record["action"],
                "flags": [],
                "reasons": [{"code": detect_result.reason or "NO_FACE", "weight": 1.0}],
                "quality": {
                    "n_faces": detect_result.n_faces,
                    "usable": False,
                    "retake_reasons": [detect_result.reason]
                },
                "signals": {},
                "cropped_face": None,
                "model_version": self.version,
                "thresholds_version": self.thresholds_version,
                "latency_ms": latency_ms,
                "input_sha256": meta.get("sha256")
            }

        primary_face = detect_result.primary_face

        # [D] Canonical Crop
        face_crop = canonical_face_crop(
            image=pil_img,
            face_box=primary_face,
            margin_ratio=self.crop_margin_ratio,
            target_size=self.target_size
        )

        # [C] Quality Gate
        quality_res = self.quality_gate.assess(face_crop=face_crop, face_box=primary_face)
        if not quality_res.is_usable:
            decision_record = route_kyc_decision(
                risk_score=0.50,
                quality_usable=False,
                quality_reasons=quality_res.reasons
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "request_id": request_id,
                "decision": decision_record["decision"],
                "route": decision_record["route"],
                "risk": 0.50,
                "action": decision_record["action"],
                "flags": [],
                "reasons": [{"code": r, "weight": 1.0} for r in quality_res.reasons],
                "quality": {
                    "n_faces": detect_result.n_faces,
                    "usable": False,
                    "retake_reasons": quality_res.reasons,
                    "metrics": quality_res.metrics
                },
                "signals": {},
                "cropped_face": face_crop,
                "model_version": self.version,
                "thresholds_version": self.thresholds_version,
                "latency_ms": latency_ms,
                "input_sha256": meta.get("sha256")
            }

        # [OOD Check] Identify non-photographic / cartoon textures
        is_in_dist, ood_reasons, ood_metrics = self.ood_detector.assess_distribution(face_crop)
        if not is_in_dist:
            decision_record = route_kyc_decision(
                risk_score=0.50,
                detection_status="OOD_NON_PHOTOGRAPHIC"
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "request_id": request_id,
                "decision": decision_record["decision"],
                "route": decision_record["route"],
                "risk": 0.50,
                "action": f"Abstain: non-photographic or synthetic graphics texture ({', '.join(ood_reasons)})",
                "priority": 0.5,
                "flags": ["OOD_NON_PHOTOGRAPHIC"],
                "reasons": [{"code": r, "weight": 1.0} for r in ood_reasons],
                "quality": {
                    "n_faces": detect_result.n_faces,
                    "usable": False,
                    "retake_reasons": ood_reasons,
                    "metrics": ood_metrics
                },
                "signals": {},
                "cropped_face": face_crop,
                "explanation_heatmap": None,
                "model_version": self.version,
                "thresholds_version": self.thresholds_version,
                "latency_ms": latency_ms,
                "input_sha256": meta.get("sha256")
            }

        # [E] Evaluate Signals (Phase 2 Multi-Modal Suite)
        artifact_sample = Sample(
            frames=[face_crop],
            timestamps_ms=[0],
            capture_meta=meta
        )
        channel_sample = Sample(
            frames=[pil_img],
            timestamps_ms=[0],
            capture_meta=meta
        )

        artifact_result = self.artifact_signal.score(artifact_sample)
        channel_result = self.channel_signal.score(channel_sample)
        frequency_result = self.frequency_signal.score(artifact_sample)
        provenance_result = self.provenance_signal.score(channel_sample)

        # [F] Risk Fusion across all 4 signals
        fusion_result = self.fusion.fuse({
            "artifact": artifact_result,
            "channel": channel_result,
            "frequency": frequency_result,
            "provenance": provenance_result
        })

        # [G] Calibrated Decision Routing
        decision_record = route_kyc_decision(
            risk_score=fusion_result.calibrated_risk,
            t_low=self.t_low,
            t_high=self.t_high,
            flags=fusion_result.flags,
            quality_usable=True
        )

        explanation_heatmap = None
        if explain and face_crop is not None:
            try:
                tensor = self.artifact_signal.transform(face_crop).unsqueeze(0).to(self.artifact_signal.device)
                heatmap = self.explainer.generate_heatmap(tensor, class_idx=0)
                explanation_heatmap = self.explainer.overlay_on_image(face_crop, heatmap)
            except Exception:
                explanation_heatmap = None

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "request_id": request_id,
            "decision": decision_record["decision"],
            "route": decision_record["route"],
            "risk": decision_record["risk"],
            "action": decision_record["action"],
            "priority": decision_record.get("priority", 0.0),
            "flags": fusion_result.flags,
            "reasons": fusion_result.reasons,
            "quality": {
                "n_faces": detect_result.n_faces,
                "usable": True,
                "metrics": quality_res.metrics,
                "retake_reasons": []
            },
            "signals": fusion_result.signal_scores,
            "cropped_face": face_crop,
            "explanation_heatmap": explanation_heatmap,
            "model_version": self.version,
            "thresholds_version": self.thresholds_version,
            "latency_ms": latency_ms,
            "input_sha256": meta.get("sha256")
        }

    def predict(self, image_input):
        """
        Streamlit backward compatibility wrapper.
        Returns: (deepfake_score, version)
        """
        res = self.assess_photo(image_input)
        return res["risk"], res["model_version"]
