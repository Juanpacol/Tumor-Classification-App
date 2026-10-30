# Report: brain MRI dataset cleaning

**Dataset:** Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle), located in `src/data/Train` and `src/data/Test`.
**Classes:** glioma, meningioma, notumor, pituitary.
**Reference run date:** 2026-10-28 (`src/data/processed/dedup_20261028T234450903626Z/`).

---

## 1. Summary

The original dataset has 7,200 images (1,400 per class in Train and 400 per class in Test). After cleaning, **3,434** remain (48 %).

Cleaning was necessary because the dataset contains:

- **Copies of the same image**, often spread across Train and Test.
- **Artificially augmented images**, even inside Test.
- **The same image with two different labels.**

With these problems, any metric on the original split is inflated: the model is evaluated on images it already saw during training. The original images are **not modified**; cleaning produces a manifest stating which images to use and which group each belongs to.

---

## 2. Why this treatment was applied

### 2.1 Data leakage inflates accuracy

In 2D MRI classification, a test image that also appears in train (or that is a near-identical slice of the same patient) lets the model "remember" instead of generalize. The literature quantifies it:

- Splitting by slice instead of by patient inflated accuracy by 29 % to 55 % on four MRI datasets. With random labels, the slice-level split reached ~96 % and the subject-level split ~50 % (Yağış et al., *Scientific Reports*, 2021).
- On the Figshare dataset (one of this dataset's sources), a strict patient-level split gave a ~88 % accuracy baseline, versus the >98 % usually published (Chhetri & Kumar, IEEE GCON, 2026).
- An audit of this same Kaggle dataset found that 28.8 % of its official test set has a near-twin in train (Vangala et al., 2026). After deduplication, the accuracy of reference models dropped by about 4 points (WICA-Net-M, *Computers*, 2026).

### 2.2 The dataset has no patient identifiers

The dataset merges three public sources (Figshare, SARTAJ and Br35H) and includes no patient IDs, so a patient-level split cannot be done directly. The alternative is to **detect images that are the same or nearly the same** and treat them as a group.

### 2.3 Problems found in this copy of the dataset

| Problem | Evidence |
|---|---|
| Exact copies | Only 6,878 of 7,200 images have unique pixels |
| Same slice with other brightness/contrast | One "notumor" slice appears 37 times: 23 in Train and 14 in Test |
| Augmented images | Files `Tr-aug-me_*` (100) and `Te-aug-me_*` (103) in meningioma, added to balance the class. Their presence in Test distorts evaluation |
| Contradictory labels | Identical images filed as meningioma in one folder and as pituitary or glioma in another |
| Train/Test leakage | 758 groups of near-identical images had members in both splits |

---

## 3. How two images are detected as "the same"

Three comparison levels are combined, from strictest to most flexible.

### 3.1 Identical pixels

Each image is decoded to 8-bit grayscale and a SHA-256 of its pixels is computed. This detects the same image even when saved with another name, format or metadata. 16-bit images are rescaled to 0–255 instead of clipped, so they do not saturate.

### 3.2 Perceptual hash (pHash): candidates

The pHash summarizes the image's low-frequency structure in 64 bits. Similar images have hashes with few differing bits (Hamming distance). It is computed on:

- the full image and the image **cropped to the brain** (without black background), because some copies have different margins and JPEG compression noise can shift the crop by a few pixels;
- each one also **mirrored** horizontally.

The smallest distance among these comparisons is used. Pairs with distance **≤ 6** become candidates.

### 3.3 Thumbnail correlation: confirmation

pHash alone is not enough: on dark, low-contrast T1 images it gives false positives between different slices. Each candidate is therefore confirmed with the **Pearson correlation between normalized 32×32 thumbnails**. This measure does not change with brightness or contrast, so a "re-windowed" copy scores ~1.0.

| Correlation | Interpretation | Action |
|---|---|---|
| ≥ 0.98 | Same slice re-processed (duplicate) | Only one image is kept |
| 0.95 – 0.98, same class | Neighbouring slices, almost always from the same patient | All are kept, in the same group |
| 0.90 – 0.95 | Uncertain zone | Listed in `review_pairs.csv` for manual review |

### 3.4 Threshold calibration

Thresholds were set on this dataset by measuring and visually reviewing sample pairs:

| pHash distance | Pairs with different class | Different-class pairs with correlation ≥ 0.95 |
|---|---|---|
| 0 | 0.2 % | 100 % (real labeling errors) |
| 2 | 0.5 % | 55 % |
| 4 | 2.4 % | 4 % |
| 6 | 12 % | 0 % |
| 8 | 31 % | 0 % |

- A true duplicate almost never changes class, so the rise in different-class pairs from distance 6 onward signals pHash false positives.
- Correlation ≥ 0.95 removes those false positives: at distance 6 no different-class pair reaches it.
- Using pHash alone at distance 6 chained different images into a single group of 2,031. With correlation confirmation, the largest group has 11 images.

---

## 4. Cleaning rules

Images confirmed as duplicates form a **duplicate cluster**. Within each cluster:

1. **Label conflict:** if copies have different labels, only those with the majority label are kept. On a tie, the whole cluster is discarded, because the correct label cannot be determined.
2. **Augmented images:** always discarded. Augmentation must be applied during training, after splitting the data, and never in test.
3. **Representative:** among the remaining images, the highest-resolution one is kept; the others are marked as its duplicates.

Each image also receives a **`group_id`** that joins duplicates with neighbouring slices of the same class. Augmented images do not create group links, so an artificial copy cannot connect two different originals.

---

## 5. Results

### 5.1 Images per class

| Split / class | Before | After | Kept |
|---|---|---|---|
| Train / glioma | 1,400 | 992 | 71 % |
| Train / meningioma | 1,400 | 539 | 39 % |
| Train / notumor | 1,400 | 275 | 20 % |
| Train / pituitary | 1,400 | 519 | 37 % |
| Test / glioma | 400 | 354 | 89 % |
| Test / meningioma | 400 | 273 | 68 % |
| Test / notumor | 400 | 125 | 31 % |
| Test / pituitary | 400 | 357 | 89 % |
| **Total** | **7,200** | **3,434** | **48 %** |

### 5.2 Removal reasons

| Reason | Images |
|---|---|
| Duplicate of another image | 3,551 |
| Augmented image | 202 |
| Label conflict | 13 |
| **Total removed** | **3,766** |

### 5.3 Train/Test leakage

| | Groups with images in both Train and Test |
|---|---|
| Before cleaning | 758 |
| After cleaning | 263 |

The remaining 263 groups are **distinct neighbouring slices** (not copies) still spread across Train and Test. They are resolved by the leakage-free split (AB-165): every kept Train image whose `group_id` also belongs to a kept Test image is removed, 495 images in total (`docs/ml-engine/decisions.md`, D15).

---

## 6. Files

### 6.1 Code (`src/preprocessing/`)

| File | What it does |
|---|---|
| `cropping.py` | Crops a grayscale image to the box containing the brain, removing the black background |
| `deduplication/__init__.py` | Describes the process and exposes the `run` function |
| `deduplication/__main__.py` | Command line: takes the data folder and thresholds, runs the cleaning and prints the summary |
| `deduplication/discovery.py` | Lists the images in `Train/` and `Test/` with their split and class |
| `deduplication/hashing.py` | 8-bit grayscale conversion, pHash, Hamming distance and normalized thumbnails |
| `deduplication/fingerprint.py` | Computes each image's "fingerprint" (hashes and thumbnails, full and cropped), detects augmented files by name and measures the correlation between two fingerprints |
| `deduplication/matching.py` | Finds candidate pairs by pHash, confirms them by correlation and groups them into duplicate clusters and related groups |
| `deduplication/selection.py` | Applies the rules in section 4 and decides which image is kept |
| `deduplication/report.py` | Computes the summary and writes the output files |
| `deduplication/pipeline.py` | Runs all steps in order and saves the result in a new timestamped folder |

### 6.2 Tests (`src/tests/`)

| File | What it checks |
|---|---|
| `synthetic.py` | Generates synthetic MRI-like slices for the tests |
| `conftest.py` | Builds a small dataset with known duplicates (exact, resized, mirrored, augmented and contradictory-label copies) |
| `test_cropping.py` | Brain cropping |
| `test_dedup_hashing.py` | pHash, Hamming distance and 16-bit rescaling |
| `test_dedup_fingerprint.py` | Augmented-name detection, contrast- and mirror-invariant correlation, and image discovery |
| `test_dedup_matching.py` | Duplicate grouping, rejection of false candidates and that augmented images do not join groups |
| `test_dedup_selection.py` | Selection rules, majority label vote and group size |
| `test_dedup_pipeline.py` | The full run and that two runs do not overwrite each other |

### 6.3 Output (`src/data/processed/dedup_<timestamp>/`)

The `src/data/` folder is excluded from git.

| File | Contents |
|---|---|
| `manifest.csv` | One row per image: `path`, `split`, `label`, `group_id`, `group_size` (kept images in the group), `keep` (1 = use) and `reason` (`kept`, `duplicate_of:<path>`, `augmented` or `label_conflict`) |
| `summary.json` | Parameters used, before/after counts per class, removal reasons and split leakage |
| `review_pairs.csv` | Pairs with correlation between 0.90 and 0.95 that were not joined, for visual review |

---

## 7. How to run it

From `app/ml_engine`:

```bash
python -m src.preprocessing.deduplication --data-dir src/data
```

Options (defaults are the calibrated values): `--max-distance 6`, `--duplicate-correlation 0.98`, `--related-correlation 0.95`.

To use the clean data, load it with `load_samples` and split it with `split_samples` (`src/training/dataset.py`). They keep `keep == 1` rows and remove Train images that share a `group_id` with Test. Then build the cross-validation folds with `StratifiedGroupKFold`, using `group_id` as the group.

---

## 8. Limitations and next steps

- **Not equivalent to a patient-level split.** Near-identical slices are joined, but two distant slices of the same patient can end up in different groups.
- **Class imbalance.** "notumor" is left with 400 images versus 1,346 for glioma: stratify the split and use class weights or balanced metrics (balanced accuracy, macro F1).
- **Manual review pending.** `review_pairs.csv` contains 1,637 uncertain pairs.
- **External validation.** Also evaluate on a dataset from another institution, first checking by hash that it shares no images with this one.

### References

- Yağış, E. et al. (2021). *Effect of data leakage in brain MRI classification using 2D convolutional neural networks.* Scientific Reports, 11. doi:10.1038/s41598-021-01681-w
- Chhetri, L., Kumar, A. (2026). *Optimizing Deep Learning for Brain Tumor Classification: A Comparative Ablation Study of Preprocessing and Augmentation Strategies.* IEEE GCON. doi:10.1109/gcon69192.2026.11649062
- Vangala, B. P. et al. (2026). *Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification.*
- Khan, M. A. et al. (2026). *WICA-Net-M: MRI-Based Brain Tumour Classification Using a Lightweight Wavelet-Integrated Coordinate Attention Network with Frequency-Aware Learning.* Computers.
