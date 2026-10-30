# Preprocessing pipeline (agreed version)

**Date:** 2026-10-29
**Scope:** `app/ml_engine`, 2D brain MRI classification into glioma, meningioma, pituitary and no tumor.
**Boards alignment:** epic #103 (rev 4) and its issues #108-#114, #150, #165-#168. This document is the written reference for that epic; when the two disagree, update both in the same change.
**LaTeX version:** [`preprocessing-pipeline.tex`](preprocessing-pipeline.tex).

---

## 1. Summary

The pipeline has ten functional blocks, B1 to B10. They were chosen by two criteria: published evidence (Consensus searches, references in section 10) and computational and implementation cost.

- Steps that **learn from data** (thresholds, augmentation, scaler, PCA, classifier, calibration) only see training rows. The official Testing split is scored once.
- Steps that **do not learn** (load, mask, geometry, tensor) are one shared function for training and inference.
- **Excluded:** steps that add cost without evidence of a gain, such as skull stripping, CLAHE, N4, denoising and SMOTE.

---

## 2. Dataset facts

Measured on the cleaned dataset versioned in `develop` (`app/ml_engine/src/data`, manifest `processed/dedup_20261028T234450903626Z/manifest.csv`), on 2026-10-29.

| Fact | Value |
|---|---|
| Images before cleaning | 7,200 (Kaggle Brain Tumor MRI Dataset, Masoud Nickparvar) |
| Images kept after cleaning (#150) | 3,434 |
| Groups of near-duplicates with members in both Training and Testing | 263 |
| Training images in those groups (removed by B2) | 495 |
| Training after B2 | 1,830 |
| Testing (official, unchanged) | 1,109 |

| Class | Training kept | Removed by B2 | Training final | Testing |
|---|---|---|---|---|
| glioma | 992 | 142 | 850 | 354 |
| meningioma | 539 | 93 | 446 | 273 |
| notumor | 275 | 15 | 260 | 125 |
| pituitary | 519 | 245 | 274 | 357 |

**Image geometry varies by class.** The source datasets differ, and so do their image shapes:

| Class | Not square | Exactly 512x512 |
|---|---|---|
| glioma | 3.9% | 1,284 / 1,346 |
| meningioma | 13.4% | 686 / 812 |
| notumor | **72.8%** | **5 / 400** |
| pituitary | 1.7% | 835 / 876 |

This means 92.5% of tumor images (2,805 of 3,034) are 512x512, against 1.25% of notumor images. A plain resize distorts almost only one class, so the model could learn the distortion instead of the anatomy. Blocks B6 and B10 address this.

> The note in #166 says "no notumor image is 512x512" and "93% (2,810 of 3,034)". The measured values are 5 of 400 and 92.5% (2,805 of 3,034). The conclusion does not change.

---

## 3. Execution order

### 3.1 Training (offline)

| Step | Block | What happens | Learns from data | Boards |
|---|---|---|---|---|
| 1 | B1 | Deduplication: SHA-256 of pixels + pHash, confirmed by thumbnail correlation. Writes the manifest with `group_id`. | No | #150 (Done) |
| 2 | B2 | Leakage-free split: keep the official Testing folder, remove from Training every `group_id` present in Testing. | No | #165 |
| 3 | B9 | Fold assignment: `StratifiedGroupKFold` (5 folds, group = `group_id`, fixed seed) on the purged Training split. | No | #112 |
| 4 | B4 | Quality thresholds: 1st percentile of SNR and entropy over the purged Training split. | **Yes**, Training only | #165, #109 |
| 5 | B3, B4, B5, B6, B8 | Per image: load, quality gate, brain mask, geometry, tensor. Same code as inference. | No | #108, #109, #166, #111 |
| 6 | - | Features: radiomics (#110) on the mask at original resolution, frozen CNN embedding (#111) on the tensor. Cached on disk. | No | #110, #111 |
| 7 | B7 | Augmentation: k copies (default 1) of each Training image, used only as fit rows of a fold. | No, but used only in fit rows | #167 |
| 8 | B9 | Per fold: scaler + PCA (radiomics), scaler (CNN), classifier with `class_weight="balanced"`. Select by mean CV macro-F1. | **Yes**, fit rows only | #112 |
| 9 | B9 | Calibration (`CalibratedClassifierCV`, sigmoid) on the same grouped folds. | **Yes**, fit rows only | #112 |
| 10 | B9 | Single evaluation on Testing. Report: accuracy, macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, ablations. | No | #112 |
| 11 | B10 | Shortcut audit and leakage guarantees written in the report. | Yes, metadata baseline, inside the same CV | #168 |
| 12 | - | Artifact bundle and GitHub Release. | No | #114 |

### 3.2 Inference (online, per uploaded image)

```
uploaded bytes
  -> B3 load (gray float32 [0, 1])           -> UnsupportedImageError
  -> B4 quality gate (stored thresholds)     -> QualityCheckError
  -> B5 brain mask
  -> radiomics on mask ----------------------+
  -> B6 crop -> pad square -> 224            |
  -> B8 3 channels + backbone mean/std       |
  -> frozen CNN -----------------------------+
  -> stored scaler / PCA / calibrated classifier
  -> PredictionResult (#107, #113)
```

No augmentation and no refitting at inference. Every stored parameter comes from the artifact manifest (#113).

### 3.3 Diagram

```
                     TRAINING                                     INFERENCE
+---------------------------------------------------------+  +----------------+
| 1  B1 Deduplication (group_id)              #150        |  |                |
| 2  B2 Leakage-free split                    #165        |  |                |
|      Training 1,830 ------------- Testing 1,109 (held)  |  |                |
| 3  B9 StratifiedGroupKFold on Training      #112        |  |                |
| 4  B4 Calibrate thresholds on Training      #165        |  |                |
| 5  B3 load -> B4 gate -> B5 mask -> B6 geometry -> B8 <-+--+-- same code    |
| 6  radiomics + frozen CNN features (cached) #110 #111   |  |                |
| 7  B7 augmentation, fit rows only           #167        |  |   (never)      |
| 8  per fold: scaler, PCA, classifier        #112        |  |                |
| 9  calibration on the same folds            #112        |  |                |
| 10 Testing scored once + report             #112        |  |                |
| 11 B10 shortcut audit + guarantees          #168        |  |                |
| 12 artifact bundle -> Release               #114        |  |                |
+---------------------------------------------------------+  +----------------+
```

### 3.4 Why the folds come after B1 and B2

The rule "split before anything else" means before any step that learns from data, not before the cleaning.

- `StratifiedGroupKFold` needs the `group_id` that B1 creates.
- It splits the Training set that B2 defines.

Steps that do not learn (B3, B5, B6, B8) give the same output for an image whatever the split, so their position relative to the folds does not matter.

---

## 4. Blocks

Each block has one purpose, a defined input and output, and testable rules.

### B1. Deduplication

- **Input:** `Train/` and `Test/` folders. **Output:** `manifest.csv` with `path, split, label, group_id, group_size, keep, reason`.
- **Rule:** exact copies (pixel SHA-256) and near-copies (pHash distance <= 6 confirmed by thumbnail correlation >= 0.98) form one cluster, of which one image is kept. Neighbouring slices (correlation 0.95-0.98, same class) share a `group_id`. Augmented files and label conflicts are dropped. Raw images are never modified.
- **Tools:** Pillow, NumPy, SciPy. **Cost:** low, run once.
- **Evidence:** R1, R2. **Boards:** #150 (Done). Details in `app/ml_engine/DATASET_CLEANING_REPORT.md`.

### B2. Leakage-free split

- **Input:** manifest. **Output:** Training and Testing sample lists.
- **Rule:** the official Testing folder is the held-out set. Every Training image whose `group_id` also appears in Testing is removed. Removed counts are reported per class.
- **Guarantee:** no `group_id` in both splits (unit test).
- **Tools:** Python only. **Cost:** very low.
- **Evidence:** R1 (28.8% of the official test split of this dataset has a near-twin in training), R3 (slice-level splits inflate accuracy by 29 to 55 points). **Boards:** #165.

### B3. Canonical load

- **Input:** JPG/PNG bytes. **Output:** 2D `float32` array in [0, 1].
- **Rule:** format detected by content, gray conversion, 16-bit images rescaled (not clipped), size limits, `UnsupportedImageError` for anything else.
- **Tools:** Pillow, NumPy. **Cost:** low.
- **Evidence:** convention (R5). **Boards:** #108 (Done).

### B4. Quality gate

- **Input:** 2D image. **Output:** SNR and Shannon entropy, or `QualityCheckError` with the measured values.
- **Rule:** measured on the original image. Thresholds are the 1st percentile over the purged Training split only. Testing images are never read during calibration.
- **Tools:** scikit-image. **Cost:** low.
- **Evidence:** no external evidence for the thresholds; they are calibrated on our data. **Boards:** #109 (gate, Done), #165 (calibration).

### B5. Brain mask

- **Input:** 2D image. **Output:** boolean mask.
- **Rule:** Otsu threshold, largest connected component, hole filling. **No skull stripping.** An empty or implausible mask falls back to the full image without failing.
- **Tools:** scikit-image, SciPy. **Cost:** low.
- **Evidence:** skull stripping lowered accuracy with a patient-level split (R6) and had a negligible effect in R7. **Boards:** #166 (replaces the skull-stripping step of #109).

### B6. Geometry

- **Input:** image and mask. **Output:** 224x224 image.
- **Rule:** crop to the bounding box of the mask plus a small margin, pad to a square with black, resize (`INTER_AREA` to shrink, `INTER_LINEAR` to enlarge). The aspect ratio of the crop is preserved.
- **Tools:** OpenCV. **Cost:** very low.
- **Evidence:** cropping is common practice (R5, R8). The square padding fixes the class-dependent distortion measured in section 2. **Boards:** #166.

### B7. Train-only augmentation

- **Input:** a Training image after B6. **Output:** k augmented copies (default k = 1).
- **Rule:** fixed seed. Rotation +/-10 degrees, horizontal flip, scale +/-10%, brightness/contrast +/-10%. Each copy inherits the label, `group_id` and fold of its original. Copies are used only as fit rows of a fold and in the final fit; never in validation folds, Testing or inference. The feature cache key includes the transform parameters.
- **Tools:** OpenCV, NumPy (no new dependency). **Cost:** about 3 extra minutes of CPU feature extraction (1,830 images at about 90 ms).
- **Evidence:** train-only augmentation is a strong rule (R4, R2). Its gain with a frozen CNN is weak evidence (R9, R5), so a +/-augmentation ablation decides whether it stays enabled. **Boards:** #167.

### B8. CNN tensor

- **Input:** 224x224 image. **Output:** `(3, 224, 224)` `float32`.
- **Rule:** gray channel replicated to 3 channels, normalized with the mean and std of the selected backbone, read from its configuration and checked against the backbone bundle. Torchvision ImageNet backbones use the ImageNet values. A timm ResNet50V2 would use 0.5 / 0.5 (R11).
- **Tools:** NumPy. **Cost:** very low.
- **Evidence:** R5, R10, R11. **Boards:** #166, #111.

### B9. Validation protocol

- **Rule:** `StratifiedGroupKFold` (5 folds, `group_id`, fixed seed) on the purged Training split. Every fitted step (scaler, PCA, classifier, calibration) is fitted on the fit rows of each fold inside one scikit-learn `Pipeline`. Classes are balanced with `class_weight="balanced"`. Selection uses mean CV macro-F1. Testing is scored once, by the calibrated model.
- **Metrics:** accuracy, macro-F1, balanced accuracy, per-class precision and recall, confusion matrix. Expected hardest pair: glioma vs meningioma.
- **Ablations (CV macro-F1, never on Testing):** CNN-only vs radiomics-only vs hybrid, and +/-augmentation (#167).
- **Tools:** scikit-learn. **Cost:** low.
- **Evidence:** R2, R4, R9, R12. **Boards:** #112.

### B10. Shortcut audit

- **Rule:** a metadata-only baseline (width, height, file size) is evaluated with the same grouped CV, and its balanced accuracy is written in the report. The report also gives the Testing recall of notumor split by source size (512x512 vs other) and the status of every guarantee of section 6. If the baseline reaches 0.70 balanced accuracy or more, the CLI warns and the result is presented to the team before a model is published.
- **Tools:** scikit-learn. **Cost:** very low.
- **Evidence:** R1 (file-header features with no anatomy separated tumor from no tumor at 0.959 balanced accuracy). The 0.70 level is our choice. **Boards:** #168.

---

## 5. Steps with and without state

| Type | Blocks | Fitted on |
|---|---|---|
| Without state (same output for an image whatever the split) | B3, B5, B6, B8, CNN embedding, radiomic features | Nothing |
| With state | B4 thresholds | Purged Training split |
| With state | Scaler, PCA, classifier, calibration | Fit rows of each fold |
| Training only | B1, B2, B7, B9, B10 | Not used at inference |

---

## 6. Leakage guarantees

Each guarantee is a unit test or a report entry.

| # | Guarantee | Block | Boards |
|---|---|---|---|
| G1 | No `group_id` is present in both Training and Testing | B2 | #165 |
| G2 | Quality thresholds are computed from Training images only | B4 | #165 |
| G3 | No augmented copy reaches a validation fold, Testing or inference | B7 | #167 |
| G4 | An augmented copy is always in the same fold as its original | B7 | #167 |
| G5 | B3-B6 and B8 give the same tensor at training and at inference for the same image | B3-B8 | #166, #113 |
| G6 | The output of B6 is always 224x224 and keeps the aspect ratio of the crop | B6 | #166 |
| G7 | Testing is scored exactly once per run | B9 | #112 |

---

## 7. Excluded by cost/benefit

| Option | Why not |
|---|---|
| Skull stripping (morphological or HD-BET/SynthStrip) | Lowered accuracy with a patient-level split (R6), negligible effect (R7); deep tools need GPU or large models |
| CLAHE | Contradictory evidence; harmful with a patient-level split (R6) |
| N4 bias correction, histogram matching, denoising | No measurable gain (R7); images are already windowed 8-bit JPEGs |
| SMOTE | New dependency; `class_weight="balanced"` already handles the imbalance (R12) |
| albumentations, torchvision transforms v2 | OpenCV covers the four transforms; preprocessing stays independent of PyTorch |
| Keras / TensorFlow | A second framework for one model |
| ResNet50V2 (timm) | Postponed: weak evidence of a gain over the torchvision backbones, and a new dependency |
| CNN fine-tuning | Out of scope of epic #103; a frozen CNN is stable with little data (R10) |
| Augmentation of validation or test data | Not allowed: it leaks (R4) |

---

## 8. Known limitations and open decisions

- **No patient IDs.** `group_id` is a proxy. Distant slices of one patient can still fall in different groups, so metrics can be optimistic (R3).
- **B4 is not fitted per fold.** Thresholds use the whole purged Training split, so validation rows of the CV take part in a reject-only filter at the 1st percentile. The effect is small and Testing is not involved. Accepted to keep the pipeline simple.
- **Resolution shortcut remains.** B6 removes the aspect-ratio cue, but not the differences in source resolution and compression. B10 measures what is left; only an external dataset can rule it out.
- **Feature cache invalidation.** The cache key is built from the configuration. #166 and #167 must add configuration fields (such as the crop margin and the transform parameters) so that new code never reuses stale features.
- **`skull_stripped` in the contract.** `QualityMetrics.skull_stripped` (#107) stays in the contract. Whether it is fixed to `false` or removed (a backend API change) is still to be decided in #166.
- **Thresholds chosen by us:** the 0.70 shortcut warning (B10), k = 1 augmented copy (B7) and the crop margin (B6) have no external source.

---

## 9. Traceability

| Block | Boards | `app/ml_engine/src` module | State |
|---|---|---|---|
| B1 | #150 | `preprocessing/deduplication/` | Done |
| B2 | #165 | `training/dataset.py` | To do |
| B3 | #108 | `preprocessing/loading.py` | Done |
| B4 | #109, #165 | `preprocessing/quality.py`, `preprocessing/calibrate.py` | Gate done, calibration scope to do |
| B5 | #166 | foreground mask, today in `preprocessing/skull_stripping.py` | To do |
| B6 | #166 | `preprocessing/prepare.py` | To do |
| B7 | #167 | `training/` (new) | To do |
| B8 | #166, #111 | `preprocessing/prepare.py`, `deep_learning/` | Mostly done |
| B9 | #112 | `training/` | Grouped CV done, balanced accuracy to do |
| B10 | #168 | `training/report.py` | To do |

Implementation order, from the Boards dependencies: #165 and #166, then #112 (run on the real dataset), then #167, then #168.

---

## 10. References

Evidence IDs used in this document, in epic #103 and in issues #112 and #165-#168.

| ID | Reference | Blocks |
|---|---|---|
| R1 | Vangala et al. (2026). *Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification.* [Consensus](https://consensus.app/papers/scores-that-hold-benchmarks-that-leak-measuring-dataset-vangala-guda/d0911bae424c591daab2cbcc7609c7dc/) | B1, B2, B10 |
| R2 | Saifullah et al. (2025). *Reliable Brain Tumor Classification Without Metadata: A Step-by-Step Guideline with Duplicate Removal.* [Consensus](https://consensus.app/papers/reliable-brain-tumor-classification-without-metadata-a-saifullah-baldauf/1fdc0eb9ec175f06b1c421925a964869/) | B1, B7, B9 |
| R3 | Yagis et al. (2021). *Effect of data leakage in brain MRI classification using 2D convolutional neural networks.* Scientific Reports. [Consensus](https://consensus.app/papers/details/4a618dc480585c0ba2fef48846c002d9/) | B2 |
| R4 | Chakraborty (2026). *CatBoost-Stacked Heterogeneous Deep Ensembles for Explainable Multi-Class Brain Tumor MRI Classification.* [Consensus](https://consensus.app/papers/catbooststacked-heterogeneous-deep-ensembles-for-chakraborty/c545ad671e6f562d8811c1a302e37369/) | B7, B9 |
| R5 | Kang et al. (2021). *MRI-Based Brain Tumor Classification Using Ensemble of Deep Features and Machine Learning Classifiers.* Sensors. [Consensus](https://consensus.app/papers/details/bdb919749ffd50b2a2d4c51066c9e155/) | B3, B6, B7, B8 |
| R6 | Chhetri & Kumar (2026). *Optimizing Deep Learning for Brain Tumor Classification: A Comparative Ablation Study of Preprocessing and Augmentation Strategies.* IEEE GCON. [Consensus](https://consensus.app/papers/details/dc7827953aaa53fd8326bd97b27415f6/) | B5, exclusions |
| R7 | Kondrateva et al. (2022). *Negligible effect of brain MRI data preprocessing for tumor segmentation.* Biomed. Signal Process. Control. [Consensus](https://consensus.app/papers/details/70e323e3132b521baede10bf2b0b0dda/) | B5, exclusions |
| R8 | Prayogo et al. (2025). *Hybrid CNN-Based Transfer Learning Enhances Brain Tumor Classification on MRI Images.* IEEE Access. [Consensus](https://consensus.app/papers/details/c0d97f8f5b90504493d9ed1cb2a3834f/) | B6 |
| R9 | Kibriya et al. (2022). *A Novel and Effective Brain Tumor Classification Model Using Deep Feature Fusion and Famous Machine Learning Classifiers.* [Consensus](https://consensus.app/papers/a-novel-and-effective-brain-tumor-classification-model-kibriya-amin/c4be65c4eaff5f05a2c999740c5e338f/) | B7, B9 |
| R10 | Al-Sarairah et al. (2026). *Evaluating transfer learning for brain tumor MRI classification: frozen VGG16 feature extraction versus end-to-end training.* [Consensus](https://consensus.app/papers/evaluating-transfer-learning-for-brain-tumor-mri-al-sarairah-tawil/9ef7e942eedf536bad96508e3210ad34/) | B8, exclusions |
| R11 | timm, `resnetv2.py` (pretrained configurations). [GitHub](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py) | B8 |
| R12 | Deepak & Ameer (2023). *Brain tumor categorization from imbalanced MRI dataset using weighted loss and deep feature fusion.* [Consensus](https://consensus.app/papers/brain-tumor-categorization-from-imbalanced-mri-dataset-deepak-ameer/df66aa1c63f75c6c9f69f2f643c3e596/) | B9 (class balance), exclusions |

**Quality of the evidence.** Only R1, R2 and R4 control leakage explicitly. R10 splits by slice. Several sources (R4, R6, R10) are 2026 papers with few or no citations. Most image-preprocessing choices therefore rest on convention, while the split and fold rules rest on strong evidence.

**Internal sources**

- `app/ml_engine/DATASET_CLEANING_REPORT.md`: cleaning method and results (B1).
- `docs/research/preprocessing-2d-mri-cnn-ml.md` and `docs/research/consensus-raw-preprocessing-2d-mri.md`: earlier literature review.
- `docs/ml-engine/decisions.md`: decision log of the implemented modules.
- Measurements of section 2, run on 2026-10-29 on the manifest `dedup_20261028T234450903626Z`.
- Consensus research thread for this pipeline: <https://consensus.app/search/evidence-based-preprocessing-pipeline-for-a-hybrid/Sff9ZCR3T36MdOT5xwWZ3A/>
