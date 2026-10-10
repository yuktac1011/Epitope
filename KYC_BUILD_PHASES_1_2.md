# FedShield KYC module: build spec, Phases 1 and 2 (photo), forward plan for video

Status: draft v1. Written without access to the repo, the current training set or the current model's metrics. File paths for existing code (`backend/app/kyc/inference.py`, `data.py`, `decision.py`) come from the team's own notes. Everything marked VERIFY must be checked on the build machine before relying on it.

---

## 0. Goal, and what "works on any photo" can honestly mean

**Goal.** When any image is uploaded (camera photo, WhatsApp-forwarded photo, screenshot, photo of a screen, AI-generated face), the system returns a structured, calibrated, explainable decision. Video support is added later without rewriting this module.

No detector can promise correct answers on every possible photo. What we CAN promise, and test, is:

1. **Robust input handling.** Any file the user can upload produces a structured response and never an unhandled error.
2. **Always a defined outcome.** Every input maps to one of: `LOW_RISK`, `UNCERTAIN`, `HIGH_RISK`, `RETAKE`, `CANNOT_ASSESS`. The system abstains instead of guessing.
3. **Accuracy claimed per condition, with confidence intervals**, measured on an in-the-wild evaluation set the model never trained on. No single headline number.
4. **A published list of known blind spots** (section 9).

Three different questions are being asked of each image. Keep them separate in code, in the API and on slides:

| Question | Example failure | Signal family |
| --- | --- | --- |
| Is the image usable? | Blurry, dark, no face, face too small | Quality gate |
| Is the face synthetic or manipulated? | GAN/diffusion face, face swap, AI edit | Artifact classifier |
| Is this a live capture, or a capture of a capture? | Screenshot, photo of a screen, print | Capture-channel signals |

**A screenshot is not automatically a deepfake.** A screenshot of a genuine photo should be flagged as `NON_LIVE_CAPTURE` and routed to step-up when the flow requires a live selfie. A screenshot of a synthetic face should score high on the artifact signal AND carry the channel flag. The two must not be collapsed into one score.

**Security reality to state openly.** Arbitrary file upload is a low-trust channel: the attacker controls every pixel and every metadata field. Camera capture inside our own page or app (and later the video challenge) is the stronger control. This spec makes the upload path as robust as it can be, and treats it as lower trust than in-app capture.

---

## 1. Scope

| Phase | Scope | Rough effort |
| --- | --- | --- |
| 1. Make it correct and measurable | Input pipeline, face crop, quality gate, augmentations, calibration, decision thresholds, evaluation harness, tests | About one focused night for the P0 items (6 to 8 hours, 4 people) |
| 2. Make it generalize | Multi-source training, leave-one-generator-out evaluation, capture-channel signals, provenance signals, fusion with reason codes, explanations, API v1 | Several days |
| 3. Video (planned, not specified here) | Frame sampling, temporal model, liveness challenge, injection signals | Later; section 11 fixes the interfaces now |

Out of scope for Phases 1 and 2: document (ID card) verification, face matching against an ID, voice, secure-camera SDKs, adversarial-perturbation defence.

Priority tags: **P0** must exist for the demo to be honest, **P1** should, **P2** if time allows.

---

## 2. Architecture (photo path)

```
upload bytes
   |
   v
[A] safe decode            -> reject / normalize (format, size, orientation, color)
   |
   v
[B] face detect + landmarks -> none / many / one face
   |
   v
[C] quality gate           -> RETAKE (with reasons) or continue
   |
   v
[D] crop (matched to training) ----------------------------+
   |                                                        |
   v                                                        v
[E] signals (independent, each returns score + availability + reasons)
      E1 artifact classifier (spatial, CNN)                 
      E2 capture-channel (screenshot / screen replay)       
      E3 provenance / metadata (weak, optional)             
   |
   v
[F] fusion (calibrated logistic stacker, missing-signal aware)
   |
   v
[G] decision (thresholds from validation) + reason codes + heatmap
   |
   v
JSON response  (image bytes are never stored by default)
```

Every box is a separate module with its own tests. Boxes E1 to E3 share one interface (section 11) so that video signals can be added later.

### Proposed layout (new files under `backend/app/kyc/`)

```
backend/app/kyc/
  io/loader.py            # [A] safe decode and normalize
  face/detect.py          # [B] detector + landmarks
  face/crop.py            # [D] crop, margin, resize (single source of truth)
  face/quality.py         # [C] blur, brightness, size, pose
  signals/base.py         # Signal protocol + SignalResult
  signals/artifact.py     # E1 (wraps the existing MobileNet model)
  signals/channel.py      # E2
  signals/provenance.py   # E3
  fusion.py               # [F]
  calibration.py          # temperature scaling / isotonic
  decision.py             # [G] (exists; rewritten to use calibrated thresholds)
  inference.py            # (exists; becomes a thin orchestrator)
  data.py                 # (exists)
  augment.py              # training-time degradations (Phase 1)
  explain.py              # Grad-CAM + reason codes (Phase 2)
  eval/build_eval_set.py  # manifest builder for the in-the-wild set
  eval/run_eval.py        # per-condition metrics + report
  config/kyc.yaml        # all thresholds, margins, limits
tests/kyc/               # mirrors the layout above
```

---

## 3. Phase 1 specification

### 3.1 [A] Safe input handling (P0)

Requirements:

- Accept JPEG, PNG, WebP. HEIC/HEIF if `pillow-heif` installs cleanly (VERIFY). Reject everything else with a clear error code.
- Enforce limits from config: max file bytes, max pixels (set `PIL.Image.MAX_IMAGE_PIXELS` to guard decompression bombs), max dimensions after load.
- Apply EXIF orientation (`ImageOps.exif_transpose`) BEFORE detection. Phone photos are frequently stored rotated.
- Convert to RGB uint8: handle palette, grayscale, CMYK, 16-bit and alpha (composite alpha onto a neutral background; record `had_alpha`).
- Animated formats: use the first frame and record `animated=true`.
- Sniff content type from bytes, not from the filename or the client header.
- Record, but do not trust, original format, dimensions, EXIF presence. These become inputs to the channel and provenance signals, and are logged only as a hash plus a small feature dict, never as raw EXIF (EXIF can contain GPS).

### 3.2 [B] Face detection (P0)

- Primary option: OpenCV YuNet via `cv2.FaceDetectorYN` (needs an ONNX model file from the OpenCV Zoo; check its licence). VERIFY on the build machine: `python -c "import cv2; print(cv2.__version__, hasattr(cv2,'FaceDetectorYN'))"`.
- Alternative: MediaPipe Face Detector / Face Landmarker via the current tasks API. VERIFY: older `mp.solutions` code may not work on current releases.
- Do not use Haar cascades except as a last-resort fallback: they miss non-frontal and low-light faces.
- Behaviour:
  - 0 faces: `CANNOT_ASSESS` with reason `NO_FACE`.
  - More than 1 face: score the largest if it is clearly dominant (area ratio threshold in config), else `CANNOT_ASSESS` with `MULTIPLE_FACES`. Always report `n_faces`.
- Landmarks are used for quality (pose, eye visibility) and, in Phase 3, for head-turn challenges.

### 3.3 [D] Crop (P0, the most common silent bug)

The crop must reproduce the **training** crop. Before writing code, answer from the training data, and write the answers into `kyc.yaml`:

1. Which detector produced the training crops?
2. Margin around the box (tight, or a ratio such as 1.2 to 1.3)?
3. Square or rectangular? Resize method (bilinear/bicubic/area)?
4. Normalization constants?

If the answer is unknown, inspect 50 training images and measure the average face-area fraction of each image, then match it. One function (`crop.py`) owns this logic and is used by BOTH training preprocessing and inference. No second implementation anywhere.

### 3.4 [C] Quality gate (P0)

Outcome `RETAKE` (not "fake") when quality is too low to judge. Metrics and defaults to tune on real captures:

| Check | Method | Output reason |
| --- | --- | --- |
| Blur | Variance of Laplacian on the face crop | `TOO_BLURRY` |
| Exposure | Mean luminance and clipped-pixel fraction | `TOO_DARK`, `OVEREXPOSED` |
| Face size | Face box short side in pixels | `FACE_TOO_SMALL` |
| Pose | Yaw/pitch estimated from landmarks | `FACE_NOT_FRONTAL` |
| Occlusion | Landmark visibility / confidence | `FACE_OCCLUDED` |

Thresholds live in config, are chosen by looking at the distribution on real phone captures, and are versioned. A retake must tell the user what to fix.

### 3.5 Training-time augmentation (P0)

Add to `data.py` / new `augment.py`. Apply the SAME degradation distribution to both classes, otherwise the model learns the degradation as the label.

| Degradation | Parameters (starting values) |
| --- | --- |
| JPEG compression | Quality 30 to 95, random; sometimes applied twice (double compression, as in social apps) |
| Down-up resize | Random scale 0.4 to 1.0 then back; random interpolation |
| Gaussian blur / motion blur | Small sigma, small probability |
| Sensor noise | Gaussian + Poisson-like grain |
| Color jitter | Brightness, contrast, saturation, hue (small) |
| Screenshot simulation | Resize to a random phone-screen size, letterbox with uniform bars, re-encode as PNG or JPEG |
| Random crop margin | Jitter of the crop margin around the configured value |

Evidence note: published work reports that such augmentation only PARTIALLY restores robustness to compression and resizing, so these changes must be judged on the in-the-wild set (3.7), not on the original test split.

### 3.6 Calibration and decision thresholds (P0)

- Fit **temperature scaling** (or isotonic regression if there is enough data) on a validation set that contains the degradations of 3.5. Never on the test set.
- Choose thresholds from operating points, not by hand:
  - `t_low`: the highest score at which no more than a target fraction of genuine users are sent to step-up (the target is a business parameter, default to be agreed).
  - `t_high`: the score above which precision of "fake" meets a target (also configurable).
- The hand-set values from the team notes (0.40 down to 0.15 to 0.20) may be used as a TEMPORARY demo setting only. They are not justified until derived from validation data.
- Express costs explicitly in config: `cost_false_accept : cost_false_reject`. This is business input; the code only reads it.

Decision table (all outputs include reason codes):

| Condition | Outcome | Route |
| --- | --- | --- |
| Quality gate fails | `RETAKE` | Ask user to retake |
| No face, multiple faces, out-of-distribution input | `CANNOT_ASSESS` | Step-up (never auto-approve) |
| Calibrated risk below `t_low` and no channel flag | `LOW_RISK` | Auto-pass |
| Between `t_low` and `t_high`, or any `NON_LIVE_CAPTURE` flag in a live-capture flow | `UNCERTAIN` | Step-up (video liveness) |
| Above `t_high` | `HIGH_RISK` | Manual fraud desk |

Queue ordering for the fraud desk: `priority = calibrated_risk * exposure_weight` where `exposure_weight` comes from account tier or requested limit. Keep it a pure function with a unit test.

### 3.7 Evaluation harness (P0; build it BEFORE changing the model)

Without this, every change is blind. Build it first, run the current model through it to get the **baseline**, then measure every later change against that baseline.

**Matrix.** Rows are content types, columns are capture channels. Each filled cell is evaluated separately.

| Content \ channel | Original camera/file | Social recompressed (e.g. WhatsApp) | Screenshot | Photo of a screen |
| --- | --- | --- | --- | --- |
| Real selfie (front camera) | required | required | required | P1 |
| Real photo, other camera | P1 | P1 | P1 | P2 |
| Real with beauty/portrait filter | required | P1 | P2 | P2 |
| GAN-generated face | required | P1 | required | P1 |
| Diffusion-generated face | required | P1 | required | P1 |
| Face-swap / manipulated real photo | P1 | P1 | P2 | P2 |
| AI-edited real photo (inpainting) | P2 | P2 | P2 | P2 |

**Minimum viable set for one night:** the "required" cells, at least 30 images per cell. Be honest about the width of the confidence interval: the 95% half-width at p = 0.5 is about +/-18 points at n = 30, +/-10 at n = 100 and +/-7 at n = 200. With 30 per cell, results are directional only. Target 200 per cell later.

**Sourcing rules.**
- Real selfies: team members and friends, with written consent, on varied phones, lighting, skin tones, backgrounds. Store outside git, encrypted, deleted after the hackathon unless consent says otherwise.
- Synthetic faces: generate with tools whose terms allow it; record the generator and version in the manifest. Do not use generators the training set already contains for the "unseen generator" cells.
- Channel variants: produce social-recompressed versions by actually sending through the app, and screenshots on real phones, not only by simulation. Use the simulator only to scale up, and validate it against the real ones.
- Hold-out discipline: the evaluation set is never used for training, threshold tuning or model selection. Use a separate tuning split.

**Manifest** (`eval/manifest.csv`): `path, content_type, channel, generator, device, consent_id, split`.

**Metrics per cell and overall.**
- AUC and EER per content-type group.
- APCER and BPCER at the chosen thresholds (ISO/IEC 30107-3 terminology).
- Rate of `RETAKE` and `CANNOT_ASSESS` per cell, and accuracy among the cases the system DID decide ("coverage vs accuracy").
- Wilson 95% confidence interval on every proportion.
- Decision-stability: re-encode each real image at JPEG quality 90, 70, 50 and report the fraction whose outcome changes.
- Latency (p50, p95, CPU model and thread count recorded).

**Output:** `reports/kyc_eval_<model_version>.json` and a one-page markdown table. A change is accepted only if the worst cell does not regress and the report is attached.

### 3.8 Tests (P0)

- Loader: truncated file, zero bytes, wrong extension, huge dimensions, CMYK JPEG, 16-bit PNG, PNG with alpha, grayscale, animated GIF/WebP, EXIF-rotated, HEIC (if enabled), non-image bytes.
- Detector: no face, tiny face, multiple faces, profile, partial occlusion.
- Crop: golden-image test comparing training-time and inference-time crops are byte-identical for the same input.
- Quality: synthetic blur/dark inputs cross thresholds in the expected direction.
- Decision: threshold edges, `RETAKE` precedence, channel-flag override, priority ordering.
- Privacy: log output contains no raw EXIF and no image bytes.

### 3.9 Phase 1 exit criteria

1. The robustness corpus (3.8) produces a structured response for every input with zero unhandled exceptions.
2. The baseline and post-change evaluation reports both exist, in the same format.
3. Training and inference share one crop function, proven by the golden test.
4. Thresholds come from a script that reads validation data, not from hand edits.
5. Quality gate returns `RETAKE` with a reason on deliberately bad inputs.
6. Latency measured and written down.

---

## 4. Phase 2 specification

### 4.1 Training data and splits (P0)

- Mix several sources so the model sees more than one generator and capture style. Candidates (VERIFY access terms and licences before downloading; some require a request form and some forbid commercial use): FaceForensics++, DFDC, Celeb-DF, a StyleGAN-faces-vs-real-faces set, plus team-generated diffusion faces.
- **Split by identity and by generator, never by image.** The same person or the same source video appearing on both sides inflates results.
- **Leave-one-generator-out (LOGO):** for each generator family, train without it and test on it. Report the table. This is the honest estimate of performance on a new generator, and it is the number to show judges alongside the in-distribution score.
- Log dataset names, counts, hashes and licences in a `DATA_CARD.md`.

### 4.2 Artifact classifier (E1)

- Start from the current MobileNetV3-Small (about 2.5M parameters, light enough for CPU and later for federated head-only updates).
- Experiments, each judged on the worst cell of the evaluation matrix and LOGO, not on average accuracy:

| Variant | Why try it | Adopt only if |
| --- | --- | --- |
| Head-only fine-tune, frozen backbone | Tiny federated updates | Worst-cell AUC within a small margin of full fine-tune |
| Full fine-tune with 3.5 augmentations | Standard baseline | Beats the Phase 1 model on the worst cell |
| ImageNet-pretrained vs face-recognition-pretrained backbone | Team hypothesis | Measured gain; note that InsightFace's pretrained models are licensed for non-commercial research only |
| Larger backbone (EfficientNet-B0 to B2 range) | Capacity | Worst-cell gain justifies the larger update size and latency |

- Recommended defaults to start from (tune): AdamW, head lr 3e-4, backbone lr 1e-4, cosine schedule, 10 to 15 epochs, batch 64, label smoothing 0.05, class-balanced sampling.
- Frequency (FFT/DCT) branch: **ablation only, not a default.** Compression and resizing suppress high-frequency cues first, and frequency artifacts can help on a known generator while hurting generalization to new ones. Add it only if it improves compressed-input and LOGO results.
- Test-time augmentation (horizontal flip, 2 to 3 crops): P1, adopt only if it improves the worst cell within the latency budget.

### 4.3 Capture-channel signals (E2)

Goal: flag screenshots and screen replays as `NON_LIVE_CAPTURE`, without calling them fakes.

Cheap features (weak individually; feed them to the fusion model, never hard-code them as rules):

| Feature | Idea | Caveat |
| --- | --- | --- |
| Container/format | PNG with no camera metadata | Messaging apps strip EXIF; absence is weak evidence |
| Dimensions | Exact match to a configurable list of device screen sizes and aspect ratios | List is populated from test devices; it will go stale |
| UI chrome | Uniform bars at top/bottom, status-bar-like regions, sharp axis-aligned borders | Cropped screenshots hide it |
| Noise residual level | Camera sensor noise is usually present in real captures | Heavy denoising and filters reduce it |
| Moire / periodic peaks | Strong off-center peaks in the 2D spectrum of the face region | Needs a real replay corpus to tune; likely noisy |
| Specular / glare patterns | Screen glare and bezel reflections | Highly variable |

Plus a small CNN trained on **real and simulated** screenshots/replays labeled by channel (real selfie vs screenshot of the same selfie), validated on the real-phone subset. Replay-attack datasets (Replay-Attack, OULU-NPU, CelebA-Spoof, etc.) exist; VERIFY licences and access first.

Honest limit: a single still image cannot defend against an attacker who controls the whole pipeline (virtual camera, injected frames). Video challenge-response (Phase 3) and in-app capture are the real answers.

### 4.4 Provenance signals (E3, P1)

- If the file carries Content Credentials (C2PA), verify them; a valid credential from a trusted tool is positive evidence of origin. `c2pa-python` is a candidate library (VERIFY).
- Absence of credentials is **neutral**, not suspicious, since almost no ordinary phone photos carry them yet.
- Cheap hints, low weight: generator names in PNG text chunks or EXIF `Software`. These are trivially removable, so they can raise suspicion but never lower it.

### 4.5 Fusion (P0)

- A logistic regression stacker over `[artifact_logit, channel_score, provenance_score, availability flags]`, calibrated on the validation set.
- Missing signals: pass an explicit availability flag and impute with the training mean; do not silently set to zero.
- Output: calibrated `risk` in [0, 1], per-signal contributions (coefficient times value), and the top reasons.
- Reason codes (stable strings, versioned): `ARTIFACT_SUSPECTED`, `NON_LIVE_CAPTURE_SUSPECTED`, `SCREENSHOT_SUSPECTED`, `SCREEN_REPLAY_SUSPECTED`, `NO_PROVENANCE`, `LOW_CONFIDENCE`, `NO_FACE`, `MULTIPLE_FACES`, `TOO_BLURRY`, and so on.

### 4.6 Uncertainty and abstention (P1)

- Out-of-distribution check: distance of the penultimate-layer embedding from the training distribution (for example Mahalanobis distance with a threshold set on validation). Inputs such as cartoons, extreme close-ups or tiny faces then yield `CANNOT_ASSESS`.
- Report "accuracy at coverage": what fraction of inputs get a decision and how accurate those decisions are, so abstaining is visible and not hidden.

### 4.7 Explanations (P1)

- Grad-CAM heatmap over the face crop for the artifact signal (the `grad-cam` package, VERIFY version). Label it "regions that influenced the score", not "where the fake is": heatmaps can mislead.
- Return reason codes always; heatmap on request.

### 4.8 API v1 (P0)

`POST /kyc/photo/score` (multipart image). Response (Pydantic model, frozen once released):

```json
{
  "request_id": "uuid",
  "decision": "LOW_RISK | UNCERTAIN | HIGH_RISK | RETAKE | CANNOT_ASSESS",
  "route": "auto_pass | step_up | fraud_desk | retake",
  "risk": 0.00,
  "flags": ["NON_LIVE_CAPTURE_SUSPECTED"],
  "reasons": [{"code": "ARTIFACT_SUSPECTED", "weight": 0.00}],
  "quality": {"n_faces": 1, "blur": 0.0, "brightness": 0.0, "face_px": 0, "retake_reasons": []},
  "signals": {
    "artifact": {"available": true, "score": 0.00},
    "channel": {"available": true, "score": 0.00},
    "provenance": {"available": false, "score": null}
  },
  "model_version": "kyc-photo-1.0.0",
  "thresholds_version": "2026-10-xx",
  "latency_ms": 0,
  "input_sha256": "..."
}
```

Rules: images are processed in memory and not persisted by default; logs hold the hash and feature dict only; every response carries model and threshold versions; rate limiting and size limits on by default.

### 4.9 Federated-learning hook (P2)

Expose `get_head_weights()` / `set_head_weights()` on the artifact model, with the backbone frozen. That keeps updates in the tens of kilobytes and lets the existing Flower pipeline federate the KYC head later (split the data by source or generator across simulated banks, compare local-only vs federated on an unseen generator). Not part of Phase 2 acceptance.

### 4.10 Phase 2 exit criteria

1. LOGO table and the full evaluation matrix are produced for the final model and for the Phase 1 baseline.
2. Screenshot and screen-replay detection is measured on real-phone data, with confidence intervals.
3. Fusion model is calibrated; reason codes are returned for every decision.
4. Worst-cell performance has not regressed against the Phase 1 model; if it has, the change is reverted.
5. API schema frozen and documented; model card and data card written.

---

## 5. Compute and latency budget

State targets, then report measurements; do not publish targets as results.

- Target: p95 under 1 second per photo on a single CPU core, including decode and detection (to be measured, then adjusted).
- Model size target for federated use: head update under 100 KB.
- Heavy signals (Grad-CAM, TTA, optional extra models) run only on `UNCERTAIN` cases to keep the common path cheap.

---

## 6. Privacy and data handling

- Consent and a signed list for every real person in the evaluation set; no minors.
- Images and manifests stored outside git (`.gitignore` the data directories), encrypted at rest, deleted after the event unless consent says otherwise.
- No third-party upload of face images (no external APIs for scoring).
- Logs: hashes and aggregate features only.

---

## 7. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Crop mismatch between train and inference | Wrong scores on all phone photos | Single crop function, golden test (3.8) |
| Training set is GAN-faces only, not video face-swaps | Detects a different attack than the problem statement describes | State it; add multi-source training and LOGO (4.1) |
| Small evaluation cells | Wide confidence intervals, false confidence | Report intervals, call results directional, grow the set |
| Beauty/portrait filters on real selfies look synthetic | False positives on genuine users | Dedicated evaluation cell; consider including filtered real photos in training |
| New generators outpace the model | Misses | LOGO reporting, step-up routing, video challenge (Phase 3), federated updates |
| Upload channel fully attacker-controlled | Evasion by re-encoding, filtering, screenshotting | Treat uploads as low trust; prefer in-app capture |
| Adversarial perturbations | Targeted evasion | Out of scope; documented limitation |
| Hand-picked thresholds | Unjustified accept/reject rates | Thresholds derived from validation (3.6) |

---

## 8. Build order (one-night plan for Phase 1 P0 items)

1. Evaluation harness and a first small set (selfies, screenshots, a few generated faces). Run the CURRENT model to get the baseline. (about 2 h, split across people while others build)
2. Safe loader plus tests. (about 1 h)
3. Face detection, crop matched to training, shared with training code, golden test. (1 to 2 h)
4. Quality gate with `RETAKE`. (about 1 h)
5. Calibration plus threshold script plus decision table. (about 1.5 h)
6. Re-run the harness and write the comparison. (about 0.5 h)
7. If time remains: augmentation retrain (2 to 3 h, mostly compute), Grad-CAM (1 h).

Do not start Phase 2 items (extra backbones, frequency branch, channel CNN) until steps 1 to 6 are green.

---

## 9. Known blind spots to publish (do not hide these)

- High-quality diffusion portraits from generators not represented in training.
- Real photos with strong beauty filters or heavy retouching (false positives).
- AI-edited real photos (partial inpainting) and face swaps in stills.
- Photos of screens or prints made under good lighting, or after heavy recompression.
- Adversarially perturbed images.
- Low-resolution, black-and-white, scanned or artistic images.
- Anything an attacker injects through a virtual camera.

---

## 10. Open questions (need an answer from the team)

1. What dataset was the current KYC model trained on, how were its crops produced, and what metrics does it currently report?
2. What exactly failed in the observed case: a real selfie scored as fake, or a fake scored as real?
3. Will KYC photos come from in-app camera capture, file upload, or both? (Decides how much weight the channel signals carry.)
4. What is the acceptable share of genuine users sent to step-up, and the cost ratio of a missed fake to a false alarm? (Business inputs for thresholds.)
5. Who can supply consented real selfies and screenshots on at least 3 phone models?

---

## 11. Forward plan for video (Phase 3): interfaces to fix NOW

So that video does not force a rewrite, define these in Phase 1:

```python
# signals/base.py
from dataclasses import dataclass, field
from typing import Protocol, Optional

@dataclass
class SignalResult:
    name: str
    available: bool
    score: Optional[float]          # higher = more suspicious
    confidence: Optional[float] = None
    reasons: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

class Signal(Protocol):
    name: str
    def score(self, sample: "Sample") -> SignalResult: ...

@dataclass
class Sample:
    frames: list            # photo = list of length 1
    timestamps_ms: list[int]
    audio: Optional[bytes]  # None for photos
    capture_meta: dict      # format, dims, device hints, EXIF flags
    challenge: Optional[dict]  # None for photos
```

- A photo is a clip of length 1. Fusion already handles missing signals, so video signals slot in later.
- Planned video signals: temporal artifact pooling over sampled frames, head-turn / blink challenge-response using landmarks, response latency, pulse (rPPG) as an experimental signal, lip-sync if audio is captured, and injection indicators (virtual-camera labels, frame-timing jitter). Browser-level injection checks are weak; a native SDK with device attestation would be the production answer.
- The existing decision table, reason-code scheme, evaluation matrix (add a "video" channel column) and API versioning carry over unchanged.
- Federated training of the video model reuses the head-only pattern from 4.9.

---

## 12. Verification log (fill in on the build machine)

| Item | Status |
| --- | --- |
| `cv2.FaceDetectorYN` available in installed OpenCV; ONNX model licence | VERIFY |
| MediaPipe tasks API version and current usage | VERIFY |
| `pillow-heif` installs and decodes HEIC | VERIFY |
| `c2pa-python` installs, verifies a sample credential | VERIFY |
| `grad-cam` works with the MobileNetV3 layer chosen | VERIFY |
| Licences of every training and evaluation dataset used | VERIFY |
| InsightFace pretrained models: non-commercial research use only | Confirmed from the project's licence text; do not use for a commercial claim |
| Training crop recipe (detector, margin, resize, normalization) | UNKNOWN until inspected |
