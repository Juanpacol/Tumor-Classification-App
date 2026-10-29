# Preprocessing pipeline for 2D brain tumor MRI (CNN feature extractor + ML classifier)

**Date:** 2026-10-28
**Source:** Consensus ("pro" research agent: 181k papers retrieved → 168 eligible → 20 included) + 3 targeted searches.
**Original thread:** https://consensus.app/search/research-question-what-is-the-standard-best-eviden/kKkMB1umSNig0a4EswBD6A/
**Raw document with all verbatim citations:** [consensus-raw-preprocessing-2d-mri.md](consensus-raw-preprocessing-2d-mri.md)

---

## 1. Main conclusion

The literature converges on a **minimal, anatomy-preserving pipeline**:

> format conversion → intensity normalization → resize to the CNN input size → grayscale to 3 channels → (optional) mild denoising → frozen/fine-tuned CNN → feature vector → scaling + selection → classical classifier (SVM-RBF as baseline).

Three key ideas:

1. **What changes results most is not preprocessing but how the dataset is split.** Splitting by image/slice instead of by patient inflates accuracy by 5 to 55 points. With a patient-level split, Figshare drops to a ~88 % baseline; the >98 % figures usually published are likely inflated.
2. **Intensity normalization is the step with the best evidence.** "Aggressive" steps (skull stripping, CLAHE, N4) have contradictory or no evidence: treat them as ablation variables, not defaults.
3. **A CNN extractor + SVM works well with little data** (better than softmax on small datasets), and **fusing with radiomic features (GLCM, LBP)** gives statistically significant gains. This supports the project's hybrid architecture.

---

## 2. Recommended pipeline (step by step)

| # | Step | Recommendation | Evidence |
|---|------|----------------|----------|
| 0 | **Patient-level split** | Split *before anything else*. No patient in both train and test. Use `GroupKFold` / `StratifiedGroupKFold` with `patient_id`. | Strong |
| 1 | **Format** | `.mat` (Figshare) / DICOM / NIfTI → 2D NumPy array (float32). | Standard |
| 2 | **Deduplication** | Exact hash (SHA-256) + perceptual hash (pHash) to remove duplicates and near-duplicates across train/test, especially in Kaggle datasets. | Strong (2026 audits) |
| 3 | **Brain cropping** | Contour / extreme-points cropping to remove the black background. Cheap and preserves anatomy. | Common practice, few ablations |
| 4 | **Intensity normalization** | Per-image min-max to [0,1] or z-score. Then the normalization the CNN expects (ImageNet mean/std when using ImageNet weights). | Strong |
| 5 | **Resize + 3 channels** | 224×224 (VGG/ResNet/DenseNet/EfficientNet-B0) or 299×299 (Inception/Xception). Replicate the gray channel 3 times. | Required for pretrained CNNs |
| 6 | **Denoising (optional)** | Mild bilateral or non-local means filter. If combined with contrast enhancement, apply carefully (order matters). | Moderate |
| 7 | **CLAHE / skull stripping / N4** | **Not by default.** Evaluate as ablations with a patient-level split. | Contradictory / weak |
| 8 | **Data augmentation** | Train only, after the split (rotation ±10°, horizontal flip, shift/zoom ±10 %). | Moderate |
| 9 | **Feature extraction** | *Global average pooling* layer (or late FC layers: `fc7` in VGG16, `fc1000` in ResNet-18). | Moderate |
| 10 | **Scaling + selection** | `StandardScaler` → PCA / ANOVA / mRMR / metaheuristics (PSO, GWO). **Fit on train only** (inside a scikit-learn `Pipeline`). | Moderate |
| 11 | **Classifier** | Baseline: SVM-RBF. Compare with XGBoost, RF, KNN, logistic regression. | Moderate |

---

## 3. Evidence per preprocessing step

| Strength | Step | What the evidence says |
|----------|------|------------------------|
| Strong (9/10) | **Patient-level split** | Slice-level splitting inflated accuracy by 30 % (OASIS), 29 % (ADNI), 48 % (PPMI) and 55 % (Versilia). With random labels: ~96 % (slice split) vs 50 % (subject split) — [Yağış et al., Sci Rep 2021](https://consensus.app/papers/details/4a618dc480585c0ba2fef48846c002d9/). |
| Strong (8/10) | **Intensity normalization** | Gave the most stable representations and the best accuracy (89.7 %) among 5 scenarios; aggressive preprocessing distorts intensities and hurts — [Firdaus et al. 2026](https://consensus.app/papers/details/e891decdcdd959e08f0eedcda09bd7e1/). |
| Moderate (7/10) | **Resize 224×224×3** | Practically universal; required by pretrained CNNs — [Yonar, Sci Rep 2025](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/). |
| Moderate (6/10) | **Bilateral denoising** | Bilateral + augmentation: 93.86 %, AUC 0.991 with a patient-level split — [Chhetri & Kumar 2026](https://consensus.app/papers/details/dc7827953aaa53fd8326bd97b27415f6/). |
| Moderate (5/10) | **Data augmentation** | The single most important factor in the controlled ablation (McNemar's test) — Chhetri & Kumar 2026. |
| Contradictory (4/10) | **CLAHE** | For: 99.8 % in [BTdiagAI 2026](https://consensus.app/papers/details/e785121bf815597494cb566685804524/) (no patient-level split). Against: CLAHE + skull stripping + augmentation dropped to 82 % with a patient-level split (Chhetri & Kumar 2026). |
| Weak (3/10) | **Skull stripping** | Contradictory: [Haider 2025](https://consensus.app/papers/details/b8377315aab25193bd6e8262ee8a1859/) reports 94.85 → 98.25 % on Figshare with HD-BET; Chhetri & Kumar find it harmful; for segmentation its contribution is negligible ([Kondrateva et al. 2022](https://consensus.app/papers/details/70e323e3132b521baede10bf2b0b0dda/)). |
| Weak (2/10) | **N4 / histogram matching** | No measurable benefit; intensity equalization does not improve models (Kondrateva et al.; note: 3D segmentation study, not 2D classification). |

**Practical reading:** studies where CLAHE and skull stripping "help" usually lack patient-level splits. With a patient-level split, "destructive" methods (that remove spatial context) tend to hurt and "constructive" ones (that preserve anatomy) tend to help.

---

## 4. Feature extraction and classification

- **Layer to extract:** GAP or a late FC layer. Best for VGG16 was `fc7`; for ResNet-18, `fc1000` ([Alrikabi & Muhammad 2024](https://consensus.app/papers/details/06577a29171057e3b15ddb0a84526e5d/)). DenseNet121 with GAP → XGBoost: 98.27 %.
- **Dimensionality:** deep features are high-dimensional and redundant (512–2048+ per network). Selection/reduction helps: PCA ([Aurna et al. 2022](https://consensus.app/papers/details/d66edc8a540b51e690cfa92f89cd8442/)), RFE ([Toğaçar et al. 2020](https://consensus.app/papers/details/7c5f933e58665a1899feb73e2bfdfafe/)), LMNN + PSO/GWO (Yonar 2025).
- **Feature ensemble from several CNNs:** concatenating the best 2–3 improves significantly ([Kang et al., Sensors 2021](https://consensus.app/papers/details/bdb919749ffd50b2a2d4c51066c9e155/)).
- **Classifier:** SVM-RBF wins in most cases, especially on large datasets (Kang 2021). With little data, CNN+SVM beats CNN+softmax and needs less compute than full fine-tuning ([Deepak & Ameer 2020](https://consensus.app/papers/details/c40767a5fb865f478ce6a4c2e1de96a8/)).
- **Frozen vs fine-tuned:** both work. Frozen GoogLeNet + SVM (98.1 %) beat fine-tuned GoogLeNet + softmax (93.1 %) in [Rasool et al. 2022](https://consensus.app/papers/details/7d0f84ae0a6b5338952cb6e20877558d/). Recommendation: start frozen (fast, reproducible) and try fine-tuning as an improvement.

### Key studies

| Study | Dataset | Preprocessing | CNN extractor | Classifier | Accuracy |
|-------|---------|---------------|---------------|------------|----------|
| Deepak & Ameer 2019 | Figshare (**patient-level split**, 5-fold) | Transfer learning | GoogLeNet | SVM / KNN | 98 % |
| Deepak & Ameer 2020 | Figshare | CNN features | Custom CNN | Multiclass SVM | 95.82 % |
| Kang et al. 2021 | 3 public datasets | Resize, feature ensemble | 13 pretrained CNNs | SVM-RBF | Best overall |
| Alrikabi & Muhammad 2024 | Figshare / REMBRANDT | CLAHE | VGG16 / ResNet-18 | SVM | 95.28–98.62 % |
| Rasool et al. 2022 | 4 classes (3,460 images) | — | GoogLeNet | SVM | 98.1 % |
| Aurna et al. 2022 | 3 datasets + merged | PCA | 2-stage ensemble | Best of 5 | 99.13 % |
| Yonar 2025 | Kaggle (7,023) | Normalize, augment | DenseNet201 + LMNN + GWO | KNN | 99.64 % |
| CNN-XGBoost 2025 | T1 MRI | GAP features | DenseNet121 | XGBoost | 98.27 % |
| Safy et al. 2026 | BraTS 2020 | AGC + DnCNN | 3 fine-tuned CNNs + GLCM | RF / XGBoost / SVM | 99.05 % |
| Iqbal et al. 2024 | BraTS / TCIA / Figshare | Resize 256, normalize | Seq-ResNet + radiomics | SVM | 97.53 % |
| Chhetri & Kumar 2026 | Figshare (**patient-level split**) | Bilateral + augmentation | VGG16 | — | 93.86 % |

> Figures >98 % almost never use a patient-level split. For this project, the realistic reference on Figshare with a patient-level split is between **~88 % and ~98 %**.

---

## 5. Radiomics fusion (relevant to our hybrid architecture)

- GLCM + deep features → significant gains (p ≤ 0.05) on all metrics ([Iqbal et al. 2024](https://consensus.app/papers/hybrid-deep-spatial-and-statistical-feature-fusion-for-iqbal-qureshi/531a14f2b24055c18f8676536ef20dc3/); Safy et al. 2026).
- InceptionV3 + dominant rotated LBP, optimized with PSO ([Sharif et al. 2020](https://consensus.app/papers/details/a14e7c0400d45cdabc2bbf48224604d2/)).
- LBP/HOG fused with deep features via CCA: 93.33 % (vs 73.88 % with DCA) — **the fusion method matters**.

**Implication:** the project's decision (Radiomics + CNN) is supported by the literature. Recommended: concatenate `[CNN features | GLCM | LBP | wavelet]`, scale each block separately, then apply selection.

---

## 6. Data leakage pitfalls (the most important ones)

1. **Image/slice-level split** → inflates 5–55 points. Figshare includes `PID` (233 patients): always use it.
2. **Duplicates in Kaggle (Nickparvar):** a 2026 audit found 28.8 % of the official test set has a near-twin in train; another corpus leaks 22.3 % of its test images byte-identically; and file-header features (no anatomy) separate tumor/no-tumor at 0.959 balanced accuracy ([Vangala et al. 2026](https://consensus.app/papers/details/d0911bae424c591daab2cbcc7609c7dc/)). Deduplication lowers accuracy by ~4 points ([WICA-Net-M 2026](https://consensus.app/papers/details/cda91b10631e589aa3e00aa968ce12f5/)). BRISC 2025 **cannot serve as external validation** for Nickparvar (857 of 1,000 test images are duplicates).
3. **Augmentation before the split** → copies of the same image in train and test.
4. **Fitting scaler / PCA / feature selection on the whole dataset** → subtle leakage. Put them in a `sklearn.pipeline.Pipeline` inside cross-validation.
5. **Report** the split method and seeds, and publish the code.

---

## 7. Concrete recommendations for `app/ml_engine`

1. **`preprocessing/`**: implement the minimal pipeline (steps 1–5) as composable transforms; CLAHE, skull stripping and bilateral as flags disabled by default.
2. **Split**: `StratifiedGroupKFold` by `patient_id` on Figshare; deduplicate with pHash when using Kaggle.
3. **Baseline**: frozen ResNet50 or DenseNet121 (ImageNet weights) → GAP → `StandardScaler` → SVM-RBF.
4. **Ablations** (all with a patient-level split): ±CLAHE, ±skull stripping, ±bilateral, ±augmentation, ±radiomics, ±PCA, frozen vs fine-tuned.
5. **Metrics**: balanced accuracy, macro F1, per-class AUC, as mean ± std over ≥3 seeds.

---

## 8. Limitations of this report

- Several key studies (Chhetri & Kumar, Firdaus, BTdiagAI, Vangala) are **from 2026, conference papers or preprints with 0 citations**: recent but barely replicated evidence.
- Kondrateva et al. studies **3D segmentation**, not 2D classification; its conclusion on N4/skull stripping is extrapolated with caution.
- Few ablations were run specifically with a **frozen CNN + classical ML**; most use end-to-end fine-tuning.
- Using ImageNet mean/std with ImageNet weights is standard transfer-learning practice, not a finding of this search.
- The search used the free Consensus plan (10 results per direct search).
