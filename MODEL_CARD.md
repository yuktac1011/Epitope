# FedShield KYC Model Card: Photo Screening Pipeline v1.0.0

## Model Details
* **Organization:** CyberRakshak FedShield
* **Model Type:** Multi-Stage Hybrid Vision Pipeline (MobileNetV3-Small Backbone + Spatial/FFT Dual Branch + Captive Heuristics)
* **Parameters:** ~2.5M parameters (frozen feature extractor with calibrated linear classification head)
* **Inference Target:** Local edge & server CPU inference (mean latency ~12ms, p95 < 18ms)
* **Model Checkpoint:** `backend/app/kyc/checkpoints/best_kyc_model.pth`

## Intended Use
* **Primary Objective:** Automated pre-screening signal for digital banking and KYC onboarding to flag synthetic generative manipulation and screen-replay attacks.
* **Secondary Objective:** Priority-based risk routing to streamline genuine customer onboarding while directing anomalous submissions to enhanced verification (video challenge) or manual desk inspection.
* **Non-Goals:** Absolute biometric proof of identity; standalone automated rejection without human-in-the-loop audit.

## Multi-Stage Architecture
1. **Safe Input Ingestion (`io/loader.py`):** Enforces 10MB/50MP bounds, MIME verification, EXIF transposition, alpha compositing.
2. **Face Detection (`face/detect.py`):** YuNet ONNX neural detector (OpenCV Zoo) with dominant face area arbitration.
3. **Quality Gating (`face/quality.py`):** Evaluates variance of Laplacian (blur), luminance histogram (exposure), and face pixel dimensions. Out-of-bounds inputs return `RETAKE`.
4. **Out-of-Distribution Check (`face/ood.py`):** Analyzes Shannon color entropy and HSV skin gamut to abstain on non-photographic inputs (`CANNOT_ASSESS`).
5. **Multi-Signal Suite (`signals/`):**
   * *Artifact Signal (E1):* Temperature-scaled MobileNetV3 evaluating spatial synthetic manipulation.
   * *Capture-Channel Signal (E2):* Aspect ratio and screenshot boundary artifacts to detect replay captures.
   * *Frequency Signal:* 2D FFT azimuthal power spectrum detecting periodic checkerboard upsampling spikes.
   * *Provenance Signal (E3):* Inspection of EXIF software and container metadata for AI generator signatures.
6. **Risk Fusion & Calibrated Routing (`fusion.py` & `decision.py`):** Weighted fusion with discrete outcomes (`LOW_RISK`, `UNCERTAIN`, `HIGH_RISK`, `RETAKE`, `CANNOT_ASSESS`).
7. **Explainability (`explain.py`):** Grad-CAM visual activation heatmaps pointing to decision-driving facial features.

## Validation Metrics
* **Average Precision (AP):** 93.3%
* **ROC-AUC:** 92.6%
* **Overall Accuracy:** 83.45%
* **Hardware Profile:** ~12ms per image on modern x86 CPU.

## Known Limitations & Blind Spots
* Extreme camera motion blur or sensor degradation may fail the quality gate.
* Real photos subjected to aggressive social beauty filters (e.g. face-smoothing algorithms) may produce elevated artifact scores.
* Novel, unseen diffusion generators with zero spatial artifacts require second-stage active video challenge (Phase 3).
