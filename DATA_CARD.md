# FedShield KYC Data Card: Training & Evaluation Splits

## Dataset Overview
* **Source:** HuggingFace Hub: `ThinothW/Deepfake-Identity-Isolated-Dataset-PreP`
* **Content:** Identity-isolated facial image pairs (genuine human captures paired with synthetically manipulated face swaps).
* **Crop Distribution:** 260×260 square bounding boxes centered on facial landmarks.
* **Format:** Uncompressed PNG/JPEG image matrices.

## Partitioning Strategy
1. **Local Training Split:** 20,000 samples balanced 50:50 between real and manipulated identities.
2. **Local Validation Split:** 2,000 samples reserved strictly for temperature scaling calibration and threshold determination.
3. **Federated Multi-Tenant Split:** Partitioned into 3 disjoint subsets across simulated institutional nodes (`Bank A`, `Bank B`, `Bank C`) without sharing raw samples.

## Preprocessing & Data Augmentation
* **Canonical Face Normalization:** Clamped 1.25× expansion resized to 224×224 via bilinear resampling.
* **Data Augmentations:**
  * Horizontal random flip ($p=0.5$).
  * Random Color Jittering ($p=0.5$: brightness $\pm0.2$, contrast $\pm0.2$, saturation $\pm0.2$, hue $\pm0.05$).
  * Random Gaussian Blur ($p=0.3$: kernel $3\times3$, $\sigma \in [0.1, 1.2]$).
  * ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

## Privacy, Consent & Governance
* **Biometric Handling:** No customer raw biometric data or EXIF GPS metadata is persisted to disk during evaluation or inference.
* **Federated Privacy:** Model parameter updates are computed locally within client actors; only head weights are exchanged across institutions via Flower FedAvg.
