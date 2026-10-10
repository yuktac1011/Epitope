# FedShield — Updated Build Specification
## Federated Transaction Fraud + Federated Image Deepfake Screening for KYC

**Project:** FedShield  
**Primary problem statement:** CSB-04 — Privacy-Preserving Fraud Signal Sharing Across Banks  
**KYC extension inspired by:** CSB-02 — Deepfake-Resistant Identity Verification for Digital Onboarding  
**Version:** Updated build plan — image-based KYC detection + federated training  
**Primary objective:** Make the KYC deepfake detector a working image-screening feature, then train the same lightweight model across simulated clients using federated learning.

> **Scope boundary:** The PaySim transaction-fraud model and the image-based KYC deepfake model are separate models trained with separate datasets and separate federated experiments. They can share an overall FedShield dashboard and priority-routing concept, but their weights and training data must not be mixed.

---

## 1. Executive decision

Build this in the following order:

1. **Image classifier:** Train and evaluate a lightweight real-vs-fake face-image classifier using transfer learning.
2. **KYC screening UI:** Let a user upload a face image and receive a deepfake-risk score, model version and review recommendation. Inference runs locally in the app process.
3. **Federated KYC experiment:** Partition the KYC training split among three simulated clients, train locally, aggregate model updates using Flower/FedAvg, and compare against local-only models.
4. **Evidence:** Record per-client metrics, communication cost and checks showing that image bytes and source records are not included in outgoing model-update messages.

Do **not** begin with a large video model, audio detector, liveness system, or fully distributed banking deployment. Image-based classification is the fastest defensible slice of the Mastercard use case. The prototype does not by itself solve real-time virtual-camera injection or prove liveness.

### The research question

> Can a lightweight image-based deepfake detector trained collaboratively across simulated institutions improve over local-only training, while keeping source images inside each client?

The expected improvement is a hypothesis to test—not a result to promise.

---

## 2. Where the KYC model fits in FedShield

### Existing transaction-fraud branch (CSB-04)

- Dataset: `flwrlabs/fed-fraud-paysim-banks`
- Inputs: transaction features such as amount, type, step and balances
- Target: `isFraud`
- Model plan: logistic-regression baseline and PyTorch MLP
- Federated process: Flower + FedAvg
- Output: local transaction risk score and priority routing

### New KYC deepfake branch (CSB-02 extension)

- Dataset: a separate real/fake face-image dataset
- Inputs: one face image at a time (later, sampled frames from a clip)
- Target: `Fake` vs `Real`
- Model plan: MobileNetV3-Small transfer learning
- Federated process: separate Flower + FedAvg experiment
- Output: deepfake-risk screening score and local KYC review action

### Combined product experience

A single FedShield dashboard can show both branches, but each has its own model, data contract, training run, evaluation and version.

**Do not claim that the KYC image has been linked to a PaySim transaction.** The datasets have no common real identities or verified linkage.

---

## 3. Dataset decision

### Primary candidate: Identity-Isolated Deepfake Face Images Dataset

**Dataset:** `ThinothW/Deepfake-Identity-Isolated-Dataset-PreP`  
**Link:** https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP

The current dataset card describes:

- **142,837 images** in total.
- Image resolution: **260 × 260**.
- Approximately balanced classes: **71,417 real and 71,420 fake**.
- Predefined train/validation/test splits with identity-aware isolation.
- Label mapping: **`0 = Fake`, `1 = Real`**.
- Source/manipulation coverage includes face swaps and entire-face synthesis.
- Listed Hugging Face metadata licence: **CC-BY-4.0**.
- The dataset card also states **academic and research use only** and says all source datasets remain subject to their original licences.

The card reports these final split counts:

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 99,910 | 49,934 | 49,976 |
| Validation | 21,611 | 10,873 | 10,738 |
| Test | 21,316 | 10,610 | 10,706 |
| **Total** | **142,837** | **71,417** | **71,420** |

**Source:** https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP

### Important dataset caveats

1. The dataset is composed of face images derived from deepfake/manipulation sources; it is **not a native set of live KYC webcam sessions**.
2. Identity isolation reduces the risk that the same identity appears in training and test. Preserve these supplied splits; do not randomly recombine them.
3. Its dataset card lists CC-BY-4.0 in metadata but separately restricts the release to academic/research use and defers to upstream source terms. **Review the current card and source licences before use**, and include attribution.
4. The dataset viewer has shown the train split separately from the full total. Trust verified split metadata, not the train-row count displayed alone.
5. Downloading all data may take time: the card reports around **2.87 GB**. If bandwidth or disk space is limited, begin by accessing the training split and using a reproducible subset after confirming how the current dataset is packaged.

### Dataset benchmark reported by the publisher

The card reports a separate EfficientNet-B2 run with:
- Test accuracy: **96.00%**
- Test AUC: **99.30%**
- Test F1: **0.9597**
- Cross-dataset AUC on eight unseen DF40 manipulation methods: **89.01%**

These are **publisher-reported benchmark results for that model/configuration**, not results of this build and not a guarantee that MobileNetV3-Small or federated training will achieve the same metrics. Cite the dataset and its associated sources in the final submission.

### Fallback dataset

**`prithivMLmods/Deepfake-vs-Real`**  
https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real

The card reports an approximately **1.78 GB** image-classification dataset with label mapping `0 = Deepfake`, `1 = Real`, and Apache-2.0 metadata, but it is **gated** and requires users to agree to share contact information. Its upstream composition/terms should be reviewed too. Use it only if access to the primary dataset is blocked and its terms are suitable.

### Video benchmark for later validation

**Meta DFDC Preview:** https://ai.meta.com/datasets/dfdc/  
Meta describes a preview dataset of 5,000 videos; the full challenge set has 124,000 videos. Access requires an AWS account and associated IAM credentials/account setup.

This is a possible external video test later, not the first training dataset. The initial model and its federated experiment are image-based.

---

## 4. Model decision

### Recommended model: MobileNetV3-Small + transfer learning

Use `torchvision.models.mobilenet_v3_small` with ImageNet-pretrained weights and replace its last classifier layer with a two-class head (`Fake`, `Real`).

Why this model:
- Small and fast enough for a prototype.
- Official torchvision implementation.
- Around 2.54 million parameters and a published ImageNet weight-file size of about 9.8 MB.
- Much lighter to federate than a large vision transformer.
- Supports local inference and the same architecture can be trained on each simulated client.

Official reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

**Important:** ImageNet-pretrained weights are not a deepfake detector by themselves. The classifier must be fine-tuned on the selected real/fake dataset.

### Starter configuration (tune only on validation data)

| Setting | Initial choice |
|---|---|
| Input | RGB face image |
| Image size | 224 × 224 |
| Architecture | MobileNetV3-Small, pretrained on ImageNet |
| Head | 2 output classes: Fake, Real |
| Loss | `CrossEntropyLoss` |
| Optimiser | AdamW |
| Learning rate | `1e-3` for the new head; lower if unfreezing backbone layers |
| Batch size | 32; reduce to 16 or 8 if out of memory |
| First stage | Freeze backbone, train head |
| Optional second stage | Unfreeze final block and fine-tune briefly |
| Initial local epochs | 2–3 |
| Initial federated rounds | 3–5 |
| Device | CUDA GPU if available; CPU for smoke tests |

These are practical starting values, not guaranteed optimal settings.

### Why not start with a large 372 MB model?

Large-model training and full-weight aggregation can create significant communication and compute overhead. The first research question is whether federated training helps, so use one lightweight architecture across clients, measure update size and compare performance. A larger pretrained model can be an inference-only baseline later, if needed.

### Video, audio and liveness scope

This first implementation classifies a still face image. A later clip demo could sample a few frames and aggregate their scores, but that would still not be a temporal liveness model or a reliable detector for every virtual-camera injection attack. Do not claim those capabilities.

---

## 5. Required technology stack

### Required

| Category | Package | Use |
|---|---|---|
| Runtime | Python 3.11 (confirm compatibility on your machine) | Main language |
| Deep learning | `torch`, `torchvision` | MobileNetV3-Small training and inference |
| Federated learning | `flwr` | Client simulation and FedAvg |
| Data loading | `datasets` | Load Hugging Face dataset splits |
| Image handling | `Pillow` | Decode RGB images safely |
| Data utilities | `numpy`, `pandas`, `pyyaml` | Subsets, metrics output and configuration |
| Evaluation | `scikit-learn` | Average Precision, ROC-AUC, precision, recall, F1, confusion matrix |
| Dashboard | `streamlit`, `plotly` | KYC upload page and experiment results |
| Tests | `pytest` | Label mapping, inference, priority routing and payload boundary |
| Optional explanation | `grad-cam` or Captum, if compatible | A supporting heatmap; not definitive proof of forensic reasoning |
| Optional video | `opencv-python-headless` | Sample video frames after the image pipeline works |

Use a virtual environment and pin known-good versions after the smoke test. Install PyTorch and torchvision using the official selector for your OS/CUDA environment: https://pytorch.org/get-started/locally/

### Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
```

Install PyTorch/torchvision from the official PyTorch selector, then:

```bash
pip install flwr datasets pandas numpy pillow scikit-learn streamlit plotly pyyaml pytest
```

Optional heatmaps:

```bash
pip install grad-cam
```

Optional later video frame sampling:

```bash
pip install opencv-python-headless
```

### Use the current Flower project scaffold

Flower's current PyTorch quickstart uses `ClientApp`, `ServerApp`, and FedAvg. Start with its official generated app rather than old tutorials built around a different API:

```bash
pip install flwr
flwr new @flwrlabs/quickstart-pytorch
cd quickstart-pytorch
pip install -e .
flwr run . --stream
```

Reference: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html

Get the official example running first. Then swap its sample classifier and dataset loader for the KYC model and data. Pin the compatible versions in `pyproject.toml`/the lock file before you begin the real experiment.

---

## 6. End-to-end architecture

```text
               LOCAL KYC INFERENCE
      Uploaded face image in local demo
                    |
                    v
          Validate file and size limit
                    |
                    v
          RGB + resize/normalise
                    |
                    v
          Local MobileNetV3-Small
                    |
                    v
          Deepfake-risk screening score
                    |
                    v
          Local KYC decision / priority
       Review | Extra check | Continue KYC


              SEPARATE KYC FEDERATION
  Simulated Bank A      Bank B      Bank C
  local images          local       local
  local training        training    training
        \                  |          /
         \       parameters/updates  /
          +-------------------------+
                     |
                     v
           Flower coordinator / FedAvg
           validate + aggregate updates
                     |
                     v
              Global KYC model
                     |
                     v
       Return to banks → validate → deploy
```

### What stays local

- Original images and image bytes
- Any face crops or applicant data
- Local training/validation/test image examples
- Local case decisions and identity information

### What the coordinator receives in the experiment

- Model parameters/updates needed for aggregation
- Round ID and client ID
- Required training metadata (for example, sample count, if the chosen strategy uses it)
- Payload size and training status logs

The prototype should have automated tests to check that image rows and image bytes are not embedded in client messages. This shows the implementation's data flow; it does **not** prove that model updates cannot leak information.

Secure aggregation and differential privacy are future protections to evaluate, not guarantees automatically provided by Flower or FedAvg.

---

## 7. Step-by-step implementation

## Step 1 — Preserve the existing FedShield project

Create a separate KYC module. Do not replace the PaySim data/model code.

Suggested additions:

```text
fedshield/
├── src/
│   └── fedshield/
│       ├── kyc/
│       │   ├── data.py
│       │   ├── model.py
│       │   ├── train_local.py
│       │   ├── evaluate.py
│       │   ├── inference.py
│       │   └── decision.py
│       ├── kyc_federation/
│       │   ├── client_app.py
│       │   └── server_app.py
│       ├── audit/
│       │   └── payload_audit.py
│       └── app.py
├── configs/
│   └── kyc.yaml
├── reports/
│   ├── kyc_eda.md
│   └── kyc_results.csv
├── artifacts/
│   └── README.md
├── tests/
│   ├── test_labels.py
│   ├── test_preprocess.py
│   ├── test_inference.py
│   ├── test_priority.py
│   └── test_payload_boundary.py
└── LICENSES_AND_ATTRIBUTION.md
```

Do not add real customer images, large image archives or checkpoints to Git. Keep download instructions, model version and attribution instead.

## Step 2 — Load and inspect data

The primary dataset card's loading example is:

```python
from datasets import load_dataset

dataset = load_dataset(
    "ThinothW/Deepfake-Identity-Isolated-Dataset-PreP"
)

print(dataset)
print(dataset["train"].column_names)
print(dataset["train"][0]["label"])
```

Inspect the schema and actual split sizes before writing the rest of the pipeline. The card uses `image` and `label`, with `0 = Fake` and `1 = Real`. Convert this mapping explicitly if your internal model uses a different convention.

Do not assume the entire 2.87 GB dataset can fit in RAM. First load and inspect it through `datasets`; then use the image subset/iterable/loading approach supported by the actual dataset files and your hardware. Avoid decoding every image simultaneously.

## Step 3 — Start with a reproducible training subset

For quick iteration:
- Sample 8,000–12,000 images from the official **training split only**, with roughly balanced classes.
- Keep the provided validation and test splits separate.
- For rapid debugging, select a smaller validation slice from the official validation split, but never use test samples to tune the model.
- Use fixed seeds and save the selected row indices or manifest.
- If source/group IDs exist, preserve group separation. The supplied dataset says identities are isolated across its official train/validation/test splits—do not destroy that property.

After the pipeline works, scale up the training subset and evaluate against a larger slice of the official held-out test split.

## Step 4 — Train the local model before using federated learning

1. Create `MobileNet_V3_Small_Weights.DEFAULT`.
2. Replace the final layer for 2 classes.
3. Freeze the backbone initially and train the classification head.
4. Use the same RGB preprocessing for training and inference.
5. Save the best checkpoint based on validation metrics.
6. Evaluate once on the held-out test data.

Suggested preprocessing:
- RGB conversion
- Resize/crop to 224 × 224
- Convert to tensor
- Normalize using the ImageNet mean and standard deviation:
  - mean `[0.485, 0.456, 0.406]`
  - std `[0.229, 0.224, 0.225]`

Use mild, justified augmentations for training only. Avoid aggressive augmentations that remove or fabricate manipulation artefacts. Make sure the output class mapping is tested.

## Step 5 — Evaluate honestly

Report:
- Average Precision / PR-AUC
- Precision, recall and F1
- ROC-AUC
- Confusion matrix
- False-positive rate
- Inference time and checkpoint size

Select risk thresholds on validation data. The test set should be used only after model/threshold selection is frozen.

Do not promise a 96% result merely because a benchmark in the dataset card reports it for a different EfficientNet-B2 configuration. MobileNet and federated training can perform differently.

## Step 6 — Build the KYC image-screening page

In Streamlit:
1. Accept JPEG/PNG only and cap upload size.
2. Decode via Pillow, convert to RGB and reject corrupt files.
3. Apply exactly the same preprocessing as training.
4. Run inference locally in the app process.
5. Display:
   - **Deepfake-risk score**
   - **Risk band:** Review / Additional verification / Continue normal KYC
   - Model version and a brief limitations note
6. Do not store uploaded media by default.
7. Do not log filenames, image bytes, face crops or applicant identifiers.

The output is a screening signal, not proof that the applicant is genuine or fraudulent. Avoid automatic rejection solely on this model's result.

## Step 7 — Make three simulated bank clients

Only use the official training split for client training.

- Divide training records between `BankA`, `BankB` and `BankC`.
- First try a balanced/IID partition for a reliable smoke test.
- Then try a label-skew/non-IID partition, where fraud-image proportions differ by client.
- Give every client the same model architecture, feature preprocessing, label mapping and parameter order.
- Make clear that these are **simulated data partitions**, not real banks with genuine KYC data silos.

The chosen dataset does not supply actual institution IDs for KYC data. Do not describe the client partitions as natural bank partitions.

## Step 8 — Add Flower/FedAvg

Start from the current Flower quickstart and replace only its dataset/model functions.

One round should work like this:
1. Coordinator sends the current global MobileNet state to selected clients.
2. Each client trains locally on its own images.
3. Client returns model parameters and required training metadata—not image records.
4. FedAvg combines compatible updates according to the strategy.
5. Coordinator returns the new model.
6. Evaluate the new model separately on each bank's held-out local test partition.

Begin with three clients and 3 rounds, then increase to 5 rounds only if run time allows. If federated training is slower or worse, record that; do not tune on the test data.

## Step 9 — Compare local-only against federated KYC

Use the same held-out evaluation protocol for:
- Local-only model A
- Local-only model B
- Local-only model C
- Federated model evaluated on A, B and C

Required report table:

| Experiment | Bank A AP | Bank B AP | Bank C AP | Macro AP | Bytes/round |
|---|---:|---:|---:|---:|---:|
| Local only | actual result | actual result | actual result | actual result | N/A |
| Federated | actual result | actual result | actual result | actual result | actual result |

Fill every value from actual run logs. Report any bank whose performance falls after federation.

## Step 10 — Connect to priority routing

Use the model score and a validation-selected threshold to produce a local action:

- **High score:** enhanced verification or manual review before account activation.
- **Uncertain/corrupt/low-quality image:** request a new capture or additional verification.
- **Lower score:** continue the normal KYC workflow; do not infer that the applicant is genuine.

Use a synthetic case ID, score, model version, risk band and review status. Keep source media out of the audit log.

## Step 11 — Optional explanations

The problem statement asks for visual-artifact explanations. A lightweight option is a Grad-CAM heatmap to show which image regions influenced the classifier score.

Treat this as a supporting visualisation only:
- A heatmap is not a causal forensic explanation.
- It does not prove the model used valid deepfake artefacts.
- Test it on real/fake examples and present it as model attention, not proof.
- If time is short, leave it as a stretch goal after local inference and FL work.

## Step 12 — Optional clip support

Only after image classification and federation work:
- Sample a few evenly spaced frames from an approved test clip.
- Score each frame with the image model.
- Aggregate frame scores into a demo-level clip screening result.
- Keep all frames from a video in the same dataset split.
- Delete temporary frames after processing.

State clearly that frame aggregation is not a trained temporal model, audio analysis or a complete defence against real-time injection attacks.

---

## 8. Federated-learning settings and cost

### Starter FL settings

| Setting | Initial value |
|---|---|
| Clients | 3 |
| Strategy | FedAvg |
| Rounds | 3, then 5 if time permits |
| Local epochs | 1–2 |
| Batch size | 16 or 32, depending on GPU |
| Training split | Only each client's partition of official training data |
| Validation/test | Separate; never used to update the model |
| Metrics | Local Average Precision/PR-AUC and supporting metrics |
| Logging | Client ID, round, update bytes, duration and status |

### Communication estimate

The torchvision MobileNetV3-Small ImageNet weights are about 9.8 MB. A two-class head changes the exact checkpoint size, and serialized Flower messages have overhead. As a rough upper estimate, sending and receiving a full ~10 MB model for three clients over five rounds could be on the order of:

`10 MB × 2 directions × 3 clients × 5 rounds ≈ 300 MB`

Measure actual serialized payload bytes; do not use this estimate as the reported result.

### Privacy facts

- Raw images remain in each client's local dataset in the intended design.
- Standard FedAvg does not guarantee that individual updates reveal no training information.
- The first prototype can verify outgoing application messages and avoid sending media.
- Secure aggregation, differential privacy, client authentication and privacy-attack evaluation are later safeguards that need separate implementation and measurement.

In a one-machine simulation, all client datasets physically reside on the same machine. That simulates the client protocol; it does not prove real institutional separation. State this clearly in the demo.

---

## 9. Testing checklist

### Dataset/model
- [ ] Dataset source, revision, licence and upstream terms recorded.
- [ ] Class mapping `Fake` / `Real` verified by tests and visual sample checks.
- [ ] Corrupt images, missing labels and duplicates reviewed.
- [ ] Official train/validation/test splits preserved.
- [ ] Preprocessing identical in training and inference.
- [ ] Checkpoint and run configuration saved.
- [ ] Validation-selected threshold frozen before test evaluation.

### KYC UI
- [ ] Image file type and size are restricted.
- [ ] Corrupt files fail safely.
- [ ] Upload runs through local inference only.
- [ ] Output is labelled a screening score, not a definitive identity result.
- [ ] Upload contents are not saved/logged by default.
- [ ] High-risk/uncertain cases go to additional review.

### Federation
- [ ] Flower quickstart runs before customisation.
- [ ] Three clients use the same model and preprocessing.
- [ ] At least one full FedAvg round completes.
- [ ] Global checkpoint can run inference.
- [ ] Local-only and federated results are compared per client.
- [ ] Bytes/round and round duration are recorded.
- [ ] Client messages exclude image bytes and source rows.
- [ ] No claims of formal privacy protection without implemented controls.

---

## 10. Time-prioritised plan

| Priority | Work | Deliverable |
|---|---|---|
| P0 | Dataset access, licence review and schema inspection | Verified source + manifest |
| P1 | Local MobileNet training | Checkpoint + honest metrics |
| P2 | Streamlit image screen | Working KYC demo |
| P3 | Priority routing + payload audit | Review workflow + tests |
| P4 | Flower/FedAvg with three clients | Federated KYC model |
| P5 | Local-vs-federated evaluation | Per-bank comparison table |
| P6 | Grad-CAM, external video test, video frames | Only if core is stable |

If time is short, a local detector demo is better than claiming an unfinished federated system. If the team completes at least one genuine FL round and evaluates it against local-only baselines, it can claim a federated KYC proof-of-concept.

---

## 11. Acceptance criteria for the KYC extension

The KYC extension is considered a successful prototype only if it can:

1. Load and preprocess a labelled image dataset reproducibly.
2. Train/save a local classifier and produce held-out evaluation results.
3. Run local image inference through the UI.
4. Route high-risk/uncertain results to review without declaring a person fraudulent.
5. Complete a Flower FedAvg run with at least two—and preferably three—simulated clients.
6. Compare local-only and federated performance per client.
7. Report communications overhead and verify that application-level client messages do not carry image bytes or raw image records.
8. Describe limitations: image-only input, synthetic/sourced benchmark data, lack of liveness/injection coverage, and the non-guaranteed privacy of model updates.

---

## 12. Research references

### Selected dataset and model
1. Identity-Isolated Deepfake Face Images Dataset:
   https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP
2. Dataset's associated source project:
   https://github.com/thinothw
3. Fallback Deepfake-vs-Real dataset (gated):
   https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real
4. Meta DFDC dataset page (video benchmark):
   https://ai.meta.com/datasets/dfdc/

### Official model and training documentation
5. torchvision MobileNetV3-Small:
   https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
6. PyTorch transfer learning tutorial:
   https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
7. Flower PyTorch quickstart (current API):
   https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
8. Hugging Face image data loading:
   https://huggingface.co/docs/datasets/image_dataset

### Evaluation context
9. DFDC dataset/paper information:
   https://ai.meta.com/datasets/dfdc/
10. DF40: Toward Next-Generation Deepfake Detection (NeurIPS 2024):
    https://arxiv.org/abs/2406.13495
11. FaceForensics++:
    https://github.com/ondyari/faceforensics

---

## Final build decision

- Use `ThinothW/Deepfake-Identity-Isolated-Dataset-PreP` as the primary candidate after reviewing the academic/research and upstream dataset terms.
- Use a reproducible 8,000–12,000-image training subset first; keep the official validation and test splits separate.
- Train MobileNetV3-Small via transfer learning.
- Build local KYC image screening and priority routing.
- Federate that same lightweight KYC model across three simulated clients with Flower/FedAvg.
- Compare against local-only models; report real, per-client metrics and communication overhead.
- Keep the existing PaySim transaction model as a separate federation.
- Do not claim real liveness, video injection detection, linked identity-to-transaction evidence, or formal model-update privacy unless those capabilities are implemented and evaluated.
