# TruID — Identity Document & Biometric Verification Models

This repository contains the research and model-development work carried out during a **TruID internship**, focused on building deep-learning components for an identity-verification / KYC (Know Your Customer) pipeline. The project covers four independent but related computer-vision problems:

1. **Face Liveness Detection** — telling a real, physically-present face/document apart from a spoofed presentation (photo of a screen, printed photo, replay attack, etc.)
2. **Real vs. Screen Detection** — detecting whether an ID card/document image was captured from a physical card or was actually a photograph of a screen (screenshot / recapture attack).
3. **Real vs. Tampered Document Detection** — detecting whether a specific region of an ID card (e.g., the photo/data area) has been digitally tampered with.
4. **Signature Verification** — verifying whether a handwritten signature belongs to a claimed individual (genuine) or is a forgery, using deep metric learning (triplet loss embeddings).

All models are built with **TensorFlow / Keras**, use **MobileNetV2** as the core CNN backbone (for a good accuracy/latency/model-size trade-off suited to mobile/server KYC pipelines), and were trained/experimented with on **Kaggle** notebooks (GPU: Tesla P100).

> **Note:** These notebooks were authored and executed on Kaggle, so file paths (`/kaggle/input/...`) refer to Kaggle datasets that are **not included in this repository**. To rerun them, the datasets must be re-mounted/re-downloaded and paths adjusted accordingly (see [Reproducing the Notebooks](#reproducing-the-notebooks)).

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [High-Level Architecture / Pipeline](#high-level-architecture--pipeline)
- [1. face-liveness.ipynb](#1-face-livenessipynb)
- [2. real-vs-screen.ipynb](#2-real-vs-screenipynb)
- [3. real-vs-tampered.ipynb](#3-real-vs-tamperedipynb)
- [4. signature-verification.ipynb](#4-signature-verificationipynb)
- [Common Techniques Used Across Notebooks](#common-techniques-used-across-notebooks)
- [Environment & Dependencies](#environment--dependencies)
- [Reproducing the Notebooks](#reproducing-the-notebooks)
- [Results Summary](#results-summary)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Certification](#certification)

---

## Repository Structure

```
TruID/
├── README.md                              # This file
├── TruID Internship Certification.pdf     # Internship completion certificate
├── face-liveness.ipynb                    # Face/card liveness (screen-spoof) evaluation notebook
├── real-vs-screen.ipynb                   # Real vs. screen-recapture classifier (training)
├── real-vs-tampered.ipynb                 # Tampered ID card region classifier (training)
└── signature-verification.ipynb           # Signature verification via triplet-loss embeddings
```

There is no application/server code, package manifest, or CI configuration in this repository — it is a **model research & experimentation workspace** consisting solely of Jupyter notebooks (plus one PDF credential). Each notebook is self-contained: it loads a dataset, builds/loads a model, trains or evaluates it, and reports metrics.

---

## High-Level Architecture / Pipeline

These four notebooks map onto a typical **document + biometric identity verification** flow used in KYC/onboarding products such as TruID's:

```
User submits ID card photo + selfie + signature
        │
        ▼
┌───────────────────────────┐
│ 1. Real vs. Screen check  │  → Is the ID card a genuine physical card, or a photo of a screen?
│    (real-vs-screen.ipynb) │
└───────────────────────────┘
        │ pass
        ▼
┌───────────────────────────┐
│ 2. Real vs. Tampered check│  → Has the ID card's photo/data region been digitally altered?
│  (real-vs-tampered.ipynb) │
└───────────────────────────┘
        │ pass
        ▼
┌───────────────────────────┐
│ 3. Face / Card Liveness   │  → Is the face/card presented live, or a spoof (screen, print, replay)?
│    (face-liveness.ipynb)  │
└───────────────────────────┘
        │ pass
        ▼
┌───────────────────────────┐
│ 4. Signature Verification │  → Does the signature match the claimed identity (genuine vs forged)?
│(signature-verification.ipynb)│
└───────────────────────────┘
        │
        ▼
   Verified / Rejected
```

Each stage is an independent binary (or embedding-based) classifier, meaning they can be deployed as separate microservices/models in a larger verification pipeline, or ensembled.

---

## 1. face-liveness.ipynb

**Goal:** Evaluate a pretrained "screen detector" model's ability to generalize across multiple liveness datasets — card liveness, hand liveness, and face liveness — to detect spoofed ("screen") presentations vs. genuine ("real") ones.

### What it does
- Loads a pre-trained Keras model `screen_detector.h5` (trained elsewhere, likely from the `real-vs-screen.ipynb` pipeline or a prior iteration of it).
- Defines `build_dataset_from_direct_paths()` — loads images from `real/` and `fake(screen)/` folders, resizes to 224×224, converts BGR→RGB, and applies:
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) per channel to normalize local contrast.
  - **Unsharp masking** (3 passes of Gaussian blur + weighted subtraction) to sharpen texture — useful for exposing moiré patterns / screen artifacts that indicate a recapture.
  - MobileNetV2 `preprocess_input` normalization ([-1, 1] range).
- Also defines `build_dataset_cnn()` — a folder-based variant that additionally **balances classes via Keras `ImageDataGenerator` augmentation** (rotation, shift, zoom, horizontal flip) when the "fake" class has fewer samples than "real".
- Runs the loaded model against **three separate held-out test sets** to check cross-domain generalization:
  1. **Card liveness test set** (`Testing/real` vs `Testing/screen`) — 481 images.
  2. **Card liveness (variant 2)** (`Testing/real` vs `Testing/screen3`) — 40 images.
  3. **Hand liveness dataset (3 classes)** — 11,221 images (real vs screen).
  4. **Face liveness dataset (3 classes, train+test+val combined)** — 7,330 images (5134 + 1100 + 1096).
- For each test set, produces a **confusion matrix** (seaborn heatmap), **accuracy**, and a full **classification report** (precision/recall/F1 per class: `screen` vs `real`).

### Why it matters
This notebook is primarily a **generalization/robustness evaluation** — checking whether a screen-detector trained on one type of liveness data (e.g. card photos) transfers to other biometric surfaces (hands, faces). This is a critical question for a KYC product that needs one anti-spoofing gate to protect multiple capture flows (ID card capture, selfie capture, hand/liveness gestures).

---

## 2. real-vs-screen.ipynb

**Goal:** Train a binary CNN classifier from scratch (transfer learning) to distinguish a **genuine physical hand/card capture** from a **photo of a screen showing that same subject** (a "recapture" or "screen replay" spoof attack).

### Dataset
- Source: `Hand Liveness Dataset (3 classes)` on Kaggle.
- The dataset directory contains 3 class-folders; the notebook picks folder[1] (index 1, alphabetically) as the "real" class and folders[0] + folders[2] as "fake" (screen) classes — i.e., it appears to merge two spoof sub-categories into one "fake" label. It **skips folders[0] entirely when iterating** (`for folder in folders[1:]`), meaning only folders[1] and folders[2] are actually loaded per this exact loop — worth double-checking dataset folder ordering when reproducing.
- Preprocessing per image: resize to 224×224, BGR→RGB, per-channel CLAHE, and a 3-pass unsharp-mask sharpening filter (same recipe as face-liveness.ipynb).
- Class balancing: if genuine ("real") samples outnumber "fake" samples, `ImageDataGenerator` augmentation (rotation ±15°, width/height shift 10%, zoom 10%, horizontal flip) is used to synthetically expand the fake class until counts match.
- Final dataset: **17,834 images** at 224×224×3, split 80/20 train/test with stratification (`train_test_split(..., stratify=Y)`).

### Model Architecture
A **MobileNetV2 transfer-learning classifier**:

```
Input (224, 224, 3)
   │
MobileNetV2 (ImageNet weights, frozen initially)
   │
GlobalAveragePooling2D
   │
BatchNormalization
   │
Dense(64, relu)
   │
BatchNormalization
   │
Dropout(0.2)
   │
Dense(1, sigmoid)   →  P(real)
```

### Training Procedure (2-stage transfer learning)
1. **Feature-extraction phase:** MobileNetV2 backbone frozen, only the classification head trained.
   - Optimizer: Adam, `lr=1e-4`
   - Loss: binary cross-entropy
   - Callbacks: `EarlyStopping` (patience 20, on `val_accuracy`), `ReduceLROnPlateau` (factor 0.5, patience 5)
   - Result: **~99.3% best validation accuracy** after 35 epochs (auto-stopped via LR decay).
2. **Fine-tuning phase:** last 60 layers of the backbone unfrozen, retrained at a lower LR (`1e-5`) with the same callback strategy.
   - Result: **100% train accuracy / ~99.82% best validation accuracy** after 45 epochs.
3. **Final evaluation** on the held-out 20% test split — confusion matrix + classification report generated.
4. **Model saved** as `screen_fingerprint_classifier.h5`.

### Why it matters
This is the training notebook behind the "screen detector" evaluated in `face-liveness.ipynb`. It directly targets **screen-replay/recapture spoofing**, a common low-effort attack vector against remote identity verification (holding a phone/tablet up to the camera instead of presenting the real card/hand/face).

---

## 3. real-vs-tampered.ipynb

**Goal:** Detect whether a specific region of interest (ROI) on an ID card — likely the photo or personal-data area — has been **digitally tampered with** (photo-swap, data edit, splicing, etc.), as opposed to being an untouched, genuine capture.

### Dataset & ROI Extraction
- Uses a **fixed ROI crop** defined by reference coordinates on a 1920×1080 reference frame (`x=1350, y=200, w=550, h=650`), converted to ratios so it scales to any input image resolution. This targets a specific card region (consistent across the dataset's card layout) — likely the photo/personal-info panel most commonly targeted by forgery.
- `process_folder_roi()`:
  - Iterates dataset subfolders; folder is labeled `tampered` (1) if `"tampered"` appears in its name, else `real` (0).
  - Crops the ROI, resizes to 224×224, then preprocesses with **LAB color-space CLAHE** (contrast-limit the L channel only, preserving color fidelity) followed by an **unsharp/sharpening convolution kernel** (`[[0,-1,0],[-1,5,-1],[0,-1,0]]`) to emphasize edge/splice artifacts typical of tampering.
  - For the **training set only**, applies simple augmentation (horizontal flip, 90° rotation) to the minority ("real") class to balance it against the tampered class, with `sklearn.utils.resample` as an additional balancing safeguard.
- Three data splits loaded from three separate Kaggle datasets: `tamper-train-set` (train), `deptest/card_data 2` (used for both validation and test — same source dataset, an anomaly worth revisiting for a properly held-out test set).
  - Train: 1,282 ROI samples (641 tampered / 641 real, balanced).
  - Val/Test: 3,714 ROI samples each (642 tampered / 3,072 real, imbalanced — no augmentation applied since `data_type != 'train'`).

### Model Architecture — Custom FPN + CBAM on MobileNetV2 backbone
This is the most architecturally sophisticated model in the repository — a **Feature Pyramid Network (FPN)** with a **Convolutional Block Attention Module (CBAM)**, built on top of MobileNetV2 feature maps:

1. **Multi-scale feature extraction** from 4 MobileNetV2 intermediate layers:
   - `block_3_expand_relu` (56×56) — fine, low-level texture
   - `block_6_expand_relu` (28×28)
   - `block_13_expand_relu` (14×14)
   - `block_16_project` (7×7) — coarse, high-level semantics
2. **1×1 convolutions** project each feature map to a common 256-channel depth.
3. **Progressive upsampling + concatenation** merges pyramid levels (FPN-style top-down pathway).
4. **Final multi-scale fusion**: all four levels upsampled to a common resolution and concatenated, followed by a 3×3 conv to fuse them into a single 256-channel feature map.
5. **CBAM attention block** (`cbam_block`):
   - *Channel attention*: shared MLP over both global-average-pooled and global-max-pooled features → sigmoid gate → channel-wise reweighting.
   - *Spatial attention*: channel-pooled (avg + max) → 7×7 conv → sigmoid gate → spatial reweighting.
   - This lets the network learn to focus on **where** (spatially) and **what** (which channels/features) matter most for detecting tampering artifacts — well suited to a task where forgeries are often localized to small, subtle regions.
6. **Classification head**: `GlobalAveragePooling2D → Dense(128, relu) → Dense(1, sigmoid)`.

This combination (multi-scale FPN features + CBAM attention) is specifically well-suited to **forgery/tamper localization-style problems**, where the discriminative signal (e.g. a splice boundary, resampling artifact, or inconsistent lighting/texture) may only occupy a small part of the image and exist at a particular spatial scale.

### Training
- All layers set trainable end-to-end (full fine-tuning, no frozen-backbone warm-up stage — unlike the other two CNN notebooks).
- Optimizer: Adam, `lr=1e-4`; Loss: binary cross-entropy.
- Callbacks: `EarlyStopping` (patience 20, monitor `val_loss`), `ReduceLROnPlateau` (patience 5, factor 0.5), `ModelCheckpoint` saving the best model to `best_model_finetuned.h5`.
- Evaluated on the held-out test set with `model.evaluate`, plus a full classification report and confusion matrix.

---

## 4. signature-verification.ipynb

**Goal:** Verify whether two handwritten signature images belong to the **same person** (genuine) or represent a **forgery**, using deep metric learning rather than a simple binary classifier — this generalizes to unseen users at inference time, which a fixed-class classifier cannot do.

### Approach: Triplet-Loss Embedding + Logistic-Regression Verifier
This is a **two-stage pipeline**:

**Stage 1 — Learn a signature embedding space** using a triplet network:
- Each training user has a set of `genuine` and `forged` signature images (loaded from a prebuilt `signature_data.pkl` dictionary: `{user_id: {'genuine': [...], 'forged': [...]}}`).
- Users are split 80/20 into `train_ids` / `val_ids` (embedding generalizes across users never seen together at inference).
- **Preprocessing** (`preprocess_signature`, `preprocess_for_mobilenet`):
  - Otsu thresholding to binarize ink vs. background, inversion so strokes are foreground.
  - **Centering via image moments** — translates the signature so its center of mass sits in the middle of the frame, normalizing for varying pen placement.
  - Resize/crop, CLAHE contrast enhancement, `fastNlMeansDenoising` to clean scan noise.
  - Grayscale image tiled to 3 channels and passed through MobileNetV2's `preprocess_input` (so it can reuse ImageNet-pretrained convolutional filters despite being trained on RGB natural images).
- **Embedding network** (`build_embedding_network`): MobileNetV2 (`pooling='avg'`) → `Dense(256)` → `BatchNormalization` → `ReLU` → **L2-normalization** — producing a 256-D unit-norm embedding per signature, i.e. a point on a hypersphere where distance encodes similarity.
- **`HardNegativeTripletGenerator`**: a custom `keras.utils.Sequence` that, for each batch, samples:
  - **Anchor + Positive**: two genuine signatures from the same user.
  - **Negative**: either a random signature from a different user (*easy negative*) or, once an embedding model is available, the **hardest** (closest-in-embedding-space) sample among a batch of candidates from a different user (*hard-negative mining*) — this significantly sharpens the decision boundary once basic separation is learned.
- **Triplet loss**: standard margin-based triplet loss,
  `L = mean(max(‖a−p‖² − ‖a−n‖² + margin, 0))`, with `margin = 1.4`.
- **Two-phase training**:
  1. *Warm-up* (10 epochs, easy/random negatives, `lr=1e-4`).
  2. *Hard-negative mining* (20 epochs, using the in-progress embedding model to mine hard negatives, `lr=5e-5`).
  - `EarlyStopping` (patience 6, `val_loss`) and `ReduceLROnPlateau` (patience 3) used throughout.
  - Training was GPU-intensive: each hard-mining epoch took ~30–35 minutes due to on-the-fly embedding computation for negative mining.

**Stage 2 — Pairwise verification classifier**:
- `generate_pairs_for_classifier_safe()` builds a labeled dataset of **absolute embedding differences** `|e1 − e2|` for:
  - **Genuine pairs** (label 1): two embeddings from the same user's genuine signatures.
  - **Forged pairs** (label 0): embeddings from two different users (proxy for "does not match claimed identity").
  - Configurable `forged_ratio` (tested at 50% and 70%) to control class balance, reflecting the real-world prior that most verification attempts should be genuine.
- A **Logistic Regression** classifier (`sklearn`, `solver='saga'`) is trained on these `|Δembedding|` feature vectors to output a genuine/forged probability — this is a lightweight, interpretable final decision layer on top of the learned embedding space (compares favorably to a fixed-distance threshold since it can learn a data-driven decision boundary).
- **Evaluation utilities**: `compute_eer` (Equal Error Rate — the operating point where false-accept rate = false-reject rate, the standard metric for verification/biometric systems), `plot_roc` (ROC curve + AUC), and `evaluate_threshold` (confusion matrix, precision/recall at a chosen probability threshold, default 0.5).

### Inference API
Two convenience functions expose a simple verification interface:
```python
get_embedding(img_path)                         # -> 256-D L2-normalized embedding
verify_signature(reference_path, test_path, threshold=0.5)
    # -> (is_genuine: bool, probability: float)
```

### Real-World Spot Checks
The notebook closes with manual verification tests against real employee/tester signature pairs (Hasaan, Abdullah, Azhan, Haniya, Taha, etc.) — genuine-vs-genuine and genuine-vs-forged comparisons — used as a qualitative sanity check beyond the aggregate validation metrics. Results show the model correctly identifies genuine matches with high confidence (e.g. 0.98 for a genuine/genuine pair) but also reveals **some forged signatures scoring as genuine** (e.g. 0.86–0.95 probability on certain forgeries), indicating skilled forgeries remain a meaningful residual risk — an expected and realistic limitation for signature verification systems, and one worth flagging to downstream product/risk teams.

### Artifacts Saved
- `embedding_model.keras` — the trained triplet embedding network.
- `signature_classifier.pkl` — the trained logistic-regression verifier (joblib).
- `signature_data.pkl` — the source signature dataset (round-tripped for reuse).
- `train_ids.pkl` / `val_ids.pkl` — the user-ID train/val split, saved for reproducibility.

---

## Common Techniques Used Across Notebooks

| Technique | Used In | Purpose |
|---|---|---|
| **MobileNetV2 (ImageNet-pretrained)** | All 4 notebooks | Lightweight backbone, good for mobile/edge KYC deployment |
| **Transfer learning (freeze → fine-tune)** | real-vs-screen, real-vs-tampered | Leverage ImageNet features, then adapt to domain-specific artifacts |
| **CLAHE contrast enhancement** | face-liveness, real-vs-screen, real-vs-tampered, signature-verification | Normalize lighting/contrast variance from different capture devices/scans |
| **Unsharp masking / sharpening kernels** | face-liveness, real-vs-screen, real-vs-tampered | Expose fine artifacts (moiré patterns, splice edges, screen texture) that indicate spoofing or tampering |
| **Class balancing via augmentation** | real-vs-screen, real-vs-tampered | Handle imbalanced real vs. fake sample counts |
| **EarlyStopping + ReduceLROnPlateau** | All model-training notebooks | Prevent overfitting, adaptive learning-rate schedules |
| **Confusion matrix + classification report** | All notebooks | Standard binary-classification evaluation |
| **Garbage collection (`gc.collect()`) & explicit `del`** | All notebooks | Manage GPU/CPU memory given large in-memory image arrays on Kaggle's shared compute |

---

## Environment & Dependencies

The notebooks were executed on **Kaggle Notebooks** with a **Tesla P100 GPU** runtime. Core dependencies (inferred from imports):

```
tensorflow (>=2.x, Keras API)
opencv-python (cv2)
numpy
scikit-learn
matplotlib
seaborn
tqdm
joblib
```

No `requirements.txt` or environment file is present in this repository — dependencies must be inferred from the notebook imports as listed above. To reproduce locally, install:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn tqdm joblib
```

A CUDA-capable GPU is strongly recommended (MobileNetV2 fine-tuning and triplet-loss training over tens of thousands of images are computationally heavy).

---

## Reproducing the Notebooks

1. **Datasets are not included.** Each notebook references Kaggle dataset paths such as `/kaggle/input/hand-liveness-dataset-3-classes/...`, `/kaggle/input/tamper-train-set/...`, `/kaggle/input/signature-verification/signature_data.pkl`, etc. To rerun:
   - Either execute on Kaggle with the corresponding datasets attached to the notebook, or
   - Download the datasets and update the hard-coded paths (`path_real`, `path_fake`, `data_dir_train`, etc.) to local equivalents.
2. **Pretrained model dependency:** `face-liveness.ipynb` loads `screen_detector.h5`, which is the output of a screen-detection training run (conceptually produced by `real-vs-screen.ipynb`, saved there as `screen_fingerprint_classifier.h5`) — run/obtain that training notebook first if you need a matching model artifact.
3. **Long training times:** `signature-verification.ipynb`'s hard-negative mining epochs take on the order of 30+ minutes each on a P100 GPU; budget compute time accordingly, or reduce `HARDMIN_EPOCHS` / dataset size for a quick smoke test.
4. Notebooks can be opened and run top-to-bottom in Jupyter, JupyterLab, VS Code's notebook interface, or re-uploaded to Kaggle.

---

## Results Summary

| Notebook | Task | Best Reported Metric |
|---|---|---|
| `real-vs-screen.ipynb` | Real vs. screen-recapture (hand liveness data) | ~99.3% val accuracy (frozen backbone) → ~99.82% val accuracy / 100% train accuracy after fine-tuning |
| `face-liveness.ipynb` | Cross-domain generalization of the screen detector | Evaluated (not summarized numerically in-notebook — see confusion matrices) across card, hand, and face liveness test sets of 40–11,221 images |
| `real-vs-tampered.ipynb` | Tampered ID-card ROI detection (FPN + CBAM) | Evaluated via `model.evaluate` + classification report on a 3,714-sample test set |
| `signature-verification.ipynb` | Genuine vs. forged signature verification | Triplet training converged to `val_loss ≈ 0.14–0.19`; logistic-regression verifier evaluated via EER, ROC/AUC, and threshold-based precision/recall; qualitative spot checks showed strong genuine-match confidence (~0.98) with some forged signatures still misclassified as genuine |

*(Note: some cell outputs — confusion matrices and full classification reports — are large and were truncated in this documentation pass; run the notebooks directly, or extract them with `jupyter nbconvert` / `jq '.cells[N].outputs'` on the `.ipynb` JSON, to see exact per-class precision/recall/F1 numbers.)*

---

## Known Limitations & Future Work

- **No held-out production test set for tampering detection** — `real-vs-tampered.ipynb` currently uses the *same* dataset (`deptest/card_data 2`) for both validation and test, which risks overstating generalization; a genuinely independent test set should be introduced.
- **Fixed ROI coordinates** in `real-vs-tampered.ipynb` assume a consistent card layout/orientation (1920×1080 reference frame) — this will not generalize to arbitrary card types or rotated/skewed captures without a card-alignment/detection preprocessing step.
- **Dataset folder-indexing logic** in `real-vs-screen.ipynb`'s `build_dataset_cnn` (`folders[1:]`, `fake_folders = [folders[0], folders[2]]`) is somewhat fragile — it depends on OS-level alphabetical sort order of dataset subfolders and should be replaced with explicit named-folder mapping to avoid silent mislabeling if the dataset structure changes.
- **Residual forgery risk in signature verification** — the spot-check results show some forged signatures still score above typical acceptance thresholds, meaning skilled forgeries remain a real risk; consider ensembling with stroke-dynamics/pressure-based features (if capture hardware supports it) or a higher decision threshold with human review fallback for borderline scores.
- **No model-serving / API code** is present — these notebooks stop at "trained model artifact"; integrating them into a live verification pipeline (REST API, batching, monitoring, versioning) is a separate engineering effort.
- **No automated tests** exist for any of the preprocessing or inference functions.

---

## Certification

`TruID Internship Certification.pdf` is the completion certificate issued for the internship during which this work was produced.
