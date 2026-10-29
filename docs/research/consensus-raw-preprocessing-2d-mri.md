# Research question: What is the standard / best-evidence preprocessing pipeline f

**Question:** Research question: What is the standard / best-evidence preprocessing pipeline for 2D brain MRI images (slices, e.g. T1, T1-CE, T2, FLAIR; public datasets such as Figshare/Cheng 2015 (glioma, meningioma, pituitary), Kaggle Brain Tumor MRI (Nickparvar), BraTS 2D slices, Br35H) used for brain tumor classification, specifically in HYBRID pipelines where a pretrained CNN (e.g. VGG16/19, ResNet50, DenseNet121/201, EfficientNet, MobileNet, Inception, Xception) is used as a deep feature extractor and the extracted features are classified with classical machine learning (SVM, Random Forest, KNN, XGBoost, logistic regression, ELM, etc.).

Please cover, citing papers for each step:
1. Ordered preprocessing steps typically applied: file format handling (DICOM/NIfTI/.mat to 2D), skull stripping / brain extraction, cropping to brain region (contour/extreme-points cropping), noise removal (median, Gaussian, anisotropic diffusion, non-local means), bias field correction (N4), contrast enhancement (histogram equalization, CLAHE), intensity normalization (min-max, z-score, ImageNet mean/std), resizing (224x224, 299x299 etc.) and grayscale-to-3-channel conversion, data augmentation (and whether it is used when the classifier is classical ML), class balancing.
2. Which of these steps have empirical evidence (ablations) showing they improve accuracy, and which are optional or harmful (e.g. CLAHE, skull stripping effect on accuracy).
3. Data-leakage pitfalls: patient-level vs slice-level splitting, augmentation before splitting, duplicates across train/test in Kaggle datasets, and how reported accuracies change.
4. Feature extraction specifics: which layer to extract (global average pooling, fully-connected layers), feature dimensionality, feature fusion from multiple CNNs, feature normalization/scaling before ML, dimensionality reduction/feature selection (PCA, LDA, ReliefF, mRMR, metaheuristics), and fine-tuning vs frozen ImageNet weights before extraction.
5. Which CNN + ML classifier combinations report the best performance and on which datasets, with accuracy figures.
6. Optionally, fusion with handcrafted radiomic/texture features (GLCM, LBP, wavelet) since our system is hybrid radiomics + CNN.

Desired output: a structured synthesis with a recommended consensus pipeline (step by step), a table of key studies (dataset, preprocessing, CNN extractor, classifier, accuracy), evidence strength for each preprocessing step, and recommendations/pitfalls. Prefer peer-reviewed papers from roughly 2017 onward.

# Best-Evidence **Preprocessing Pipeline** for Hybrid CNN + Classical-ML Brain Tumor MRI Classification

The literature converges on a **minimal, structure-preserving pipeline**: format conversion → intensity normalization → resize to the extractor's native input → grayscale-to-3-channel → (optional) denoising → CNN feature extraction → scaling/selection → classical classifier. Aggressive steps (skull stripping, CLAHE) are **contested and often harmful**, and **patient-level splitting is the single most important methodological safeguard** — slice-level splits inflate accuracy by 5–55 points.

## Evidence Strength for Each Preprocessing Step

| Evidence Strength | Step & Verdict |
|---|---|
| Evidence strength: Strong (9/10) | **Patient-level (not slice-level) splitting** — the best-replicated finding in the corpus; slice-level CV inflates accuracy by 30–55% [[1.1]](https://consensus.app/papers/deep-learning-in-brain-mri-effect-of-data-leakage-due-to-yağış-atnafu/81d7bf2bbf695066a02e65ac2c7c4224/) [[2.1]](https://consensus.app/papers/effect-of-data-leakage-in-brain-mri-classification-using-2d-yağış-atnafu/4a618dc480585c0ba2fef48846c002d9/) [[3.1]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) |
| Evidence strength: Strong (8/10) | **Intensity normalization** — most stable, discriminative representation; highest accuracy (89.70%) among five preprocessing scenarios [[4.1]](https://consensus.app/papers/analyzing-the-impact-of-preprocessing-on-deep-firdaus-muflikhah/e891decdcdd959e08f0eedcda09bd7e1/) [[4.2]](https://consensus.app/papers/analyzing-the-impact-of-preprocessing-on-deep-firdaus-muflikhah/e891decdcdd959e08f0eedcda09bd7e1/); min-max to [0,1] is standard [[5.1]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) |
| Evidence strength: Moderate (7/10) | **Resize to extractor input (224×224×3) + grayscale-to-3-channel** — near-universal and required for pretrained CNNs [[5.2]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) [[6.1]](https://consensus.app/papers/hybrid-deep-spatial-and-statistical-feature-fusion-for-iqbal-qureshi/531a14f2b24055c18f8676536ef20dc3/) |
| Evidence strength: Moderate (6/10) | **Denoising (bilateral/Gaussian)** — bilateral filtering with augmentation reached 93.86% and AUC 0.991 [[3.2]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) [[3.3]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/); non-local means used in some pipelines [[7.1]](https://consensus.app/papers/mribased-brain-tumor-classification-using-ensemble-of-kang-ullah/bdb919749ffd50b2a2d4c51066c9e155/) |
| Evidence strength: Moderate (5/10) | **Data augmentation** — most important single factor in controlled ablation [[3.4]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/); applied even with classical classifiers [[5.3]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) |
| Evidence strength: Moderate (4/10) | **CLAHE** — contested: best in one study (99.8%) [[8.1]](https://consensus.app/papers/btdiagai-a-web‐deployed-hybrid-framework-for-brain-tumor-ahmmed-antik/e785121bf815597494cb566685804524/) but detrimental when combined with augmentation (82%) [[3.3]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) |
| Evidence strength: Weak (3/10) | **Skull stripping** — mixed: beneficial in some papers, detrimental in others [[3.5]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/); negligible for segmentation [[9.1]](https://consensus.app/papers/negligible-effect-of-brain-mri-data-preprocessing-for-kondrateva-druzhinina/70e323e3132b521baede10bf2b0b0dda/) |
| Evidence strength: Weak (2/10) | **Bias-field correction (N4), histogram matching** — no measurable benefit; intensity equalization does not improve models [[9.2]](https://consensus.app/papers/negligible-effect-of-brain-mri-data-preprocessing-for-kondrateva-druzhinina/70e323e3132b521baede10bf2b0b0dda/) [[9.3]](https://consensus.app/papers/negligible-effect-of-brain-mri-data-preprocessing-for-kondrateva-druzhinina/70e323e3132b521baede10bf2b0b0dda/) |

**Figure 1:** Evidence strength for each preprocessing step in hybrid CNN-ML pipelines

## Recommended Consensus Pipeline (Step by Step)

1. **Format handling.** Convert `.mat`/DICOM/NIfTI to 2D arrays; expand dimensions and store as NumPy arrays [[10.1]](https://consensus.app/papers/accurate-brain-tumor-detection-using-deep-convolutional-khan-rahman/be5d8c9ef92c5c3087260c2db015ac86/) [[10.2]](https://consensus.app/papers/accurate-brain-tumor-detection-using-deep-convolutional-khan-rahman/be5d8c9ef92c5c3087260c2db015ac86/).
2. **Intensity normalization.** Min-max to [0,1] or z-score; this is the step with the clearest empirical support [[5.1]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) [[4.1]](https://consensus.app/papers/analyzing-the-impact-of-preprocessing-on-deep-firdaus-muflikhah/e891decdcdd959e08f0eedcda09bd7e1/).
3. **Resize + 3-channel.** Resize to 224×224×3 (or 299×299 for Inception) to match pretrained input [[5.2]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) [[6.1]](https://consensus.app/papers/hybrid-deep-spatial-and-statistical-feature-fusion-for-iqbal-qureshi/531a14f2b24055c18f8676536ef20dc3/).
4. **Optional denoising.** Bilateral or non-local means; avoid aggressive denoise-then-enhance ordering [[3.2]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) [[11.1]](https://consensus.app/papers/adaptive-multifeature-fusion-architecture-with-safy-abd-ellah/93503eb582a95fc7b0bbefeb97107693/).
5. **Skip or test skull stripping/CLAHE.** Treat as ablation variables, not defaults [[3.5]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) [[3.3]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/).
6. **Augment training set only** (rotation, flip, shift, zoom) [[5.3]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/).
7. **Patient-level split** before any augmentation or feature extraction [[3.6]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) [[2.2]](https://consensus.app/papers/effect-of-data-leakage-in-brain-mri-classification-using-2d-yağış-atnafu/4a618dc480585c0ba2fef48846c002d9/).

## Feature Extraction, Fusion, and Best-Performing Combinations

Extract from **global average pooling or late fully-connected layers**; ResNet-18's `fc1000` and VGG16's `fc7` gave the best SVM features [[12.1]](https://consensus.app/papers/deep-feature-extraction-with-svm-classifier-for-brain-alrikabi-muhammad/06577a29171057e3b15ddb0a84526e5d/). Deep features are high-dimensional and redundant, so selection matters [[5.4]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/); PCA [[13.1]](https://consensus.app/papers/a-classification-of-mri-brain-tumor-based-on-two-stage-aurna-yousuf/d66edc8a540b51e690cfa92f89cd8442/), PSO/GWO metaheuristics [[5.5]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/), and LMNN metric learning [[5.6]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) all improved separability. **SVM with RBF kernel** most often outperforms other classifiers, especially on large datasets [[7.2]](https://consensus.app/papers/mribased-brain-tumor-classification-using-ensemble-of-kang-ullah/bdb919749ffd50b2a2d4c51066c9e155/), and CNN-SVM beats softmax when data is small [[14.1]](https://consensus.app/papers/automated-categorization-of-brain-tumor-from-mri-using-cnn-deepak-ameer/c40767a5fb865f478ce6a4c2e1de96a8/).

| Study | Dataset | Preprocessing | CNN Extractor | Classifier | Accuracy |
|---|---|---|---|---|---|
| Kang et al. 2021 [[7.2]](https://consensus.app/papers/mribased-brain-tumor-classification-using-ensemble-of-kang-ullah/bdb919749ffd50b2a2d4c51066c9e155/) | Kaggle/Figshare | resize, ensemble features | 13 pretrained CNNs | SVM-RBF | best overall |
| Deepak & Ameer 2020 [[14.2]](https://consensus.app/papers/automated-categorization-of-brain-tumor-from-mri-using-cnn-deepak-ameer/c40767a5fb865f478ce6a4c2e1de96a8/) | Figshare | CNN features | custom CNN | multiclass SVM | 95.82% |
| Deepak & Ameer (GoogLeNet) [[15.1]](https://consensus.app/papers/computers-in-biology-and-medicine-deepak-ameer/fecde94069095cbe89eaec60088271cf/) | Figshare | transfer learning | GoogLeNet | SVM | 98% |
| Alrikabi & Muhammad 2024 [[12.2]](https://consensus.app/papers/deep-feature-extraction-with-svm-classifier-for-brain-alrikabi-muhammad/06577a29171057e3b15ddb0a84526e5d/) | Figshare/REMBRANDT | CLAHE | VGG16/ResNet-18 | SVM | 95.28–98.62% |
| Yonar 2025 [[5.7]](https://consensus.app/papers/a-swarm-intelligencedriven-hybrid-framework-for-brain-yonar/d05feb7e3a3a53b3b982068068ba9e57/) | Kaggle (7023) | normalize, augment | DenseNet201+LMNN+GWO | KNN | 99.64% |
| Aurna et al. 2022 [[13.2]](https://consensus.app/papers/a-classification-of-mri-brain-tumor-based-on-two-stage-aurna-yousuf/d66edc8a540b51e690cfa92f89cd8442/) | 3 datasets | PCA selection | two-stage CNN ensemble | best of 5 | 99.13% |
| CNN-XGBoost 2025 [[16.1]](https://consensus.app/papers/deep-feature-extraction-of-brain-tumor-magnetic-resonance/66ec38c08a035a09b76dd9adebdbccc2/) | T1 MRI | GAP features | DenseNet121 | XGBoost | 98.27% |
| Safy et al. 2026 [[11.2]](https://consensus.app/papers/adaptive-multifeature-fusion-architecture-with-safy-abd-ellah/93503eb582a95fc7b0bbefeb97107693/) | BraTS 2020 | AGC+DnCNN | 3 fine-tuned CNNs + GLCM | RF/XGBoost/SVM | 99.05% |
| Iqbal et al. 2024 [[6.2]](https://consensus.app/papers/hybrid-deep-spatial-and-statistical-feature-fusion-for-iqbal-qureshi/531a14f2b24055c18f8676536ef20dc3/) | BraTS/TCIA/Figshare | resize 256, normalize | Seq-ResNet + radiomics | SVM | 97.53% |
| Chhetri & Kumar 2026 [[3.3]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) | Figshare (patient-level) | bilateral+augment | VGG16 | — | 93.86% |

**Figure 2:** Key hybrid CNN-ML studies with datasets, preprocessing, extractors, classifiers and accuracy

**Radiomics fusion** consistently helps: InceptionV3 features concatenated with dominant-rotated LBP improved texture analysis [[17.1]](https://consensus.app/papers/active-deep-neural-network-features-selection-for-sharif-li/a14e7c0400d45cdabc2bbf48224604d2/); GLCM fusion with deep features produced significant gains (p ≤ 0.05) [[6.3]](https://consensus.app/papers/hybrid-deep-spatial-and-statistical-feature-fusion-for-iqbal-qureshi/531a14f2b24055c18f8676536ef20dc3/) [[11.3]](https://consensus.app/papers/adaptive-multifeature-fusion-architecture-with-safy-abd-ellah/93503eb582a95fc7b0bbefeb97107693/); LBP/HOG fused via CCA reached 93.33% [[18.1]](https://consensus.app/papers/fusion-of-handcrafted-and-deeplearning-features-for-brain-hanumanthappa-guruprakash/bf3f323b20265e4f882e24cffa218097/).

## Pitfalls and Recommendations

The dominant pitfall is **data leakage**: random image-level splits place the same patient's slices in train and test, inflating accuracy by 5–30 points [[3.1]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/) [[3.7]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/), with measured boosts of 30% (OASIS), 29% (ADNI), 48% (PPMI) and 55% (Versilia) [[1.1]](https://consensus.app/papers/deep-learning-in-brain-mri-effect-of-data-leakage-due-to-yağış-atnafu/81d7bf2bbf695066a02e65ac2c7c4224/). A randomly-labeled control reached ~96% under slice-level split versus 50% under subject-level [[1.2]](https://consensus.app/papers/deep-learning-in-brain-mri-effect-of-data-leakage-due-to-yağış-atnafu/81d7bf2bbf695066a02e65ac2c7c4224/). Under strict patient-level splitting, Figshare baseline drops to a "clinically plausible" 87.92%, suggesting published >98% figures are likely overstated [[3.8]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/); architecture degradations of up to 3.71% were observed [[19.1]](https://consensus.app/papers/attentionenhanced-swin-transformers-for-robust-brain-chhetri-datta/3685dc605c525b43a9f15b37fdd739f2/).

Additional cautions: augmentation must be applied **after** splitting and only to training data [[3.6]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/); aggressive preprocessing distorts intensity distributions and spatial cues [[4.2]](https://consensus.app/papers/analyzing-the-impact-of-preprocessing-on-deep-firdaus-muflikhah/e891decdcdd959e08f0eedcda09bd7e1/); and the "destructive vs. constructive" taxonomy favors context-preserving methods [[3.9]](https://consensus.app/papers/optimizing-deep-learning-for-brain-tumor-classification-a-chhetri-kumar/dc7827953aaa53fd8326bd97b27415f6/). Report the split method explicitly and release code [[2.3]](https://consensus.app/papers/effect-of-data-leakage-in-brain-mri-classification-using-2d-yağış-atnafu/4a618dc480585c0ba2fef48846c002d9/) [[2.4]](https://consensus.app/papers/effect-of-data-leakage-in-brain-mri-classification-using-2d-yağış-atnafu/4a618dc480585c0ba2fef48846c002d9/).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
`1.` Yağış E, Atnafu S, De Herrera AGS, et al. Deep Learning in Brain MRI: Effect of Data Leakage Due to Slice-level Split Using 2D Convolutional Neural Networks. 2021. doi:10.21203/rs.3.rs-464091/v1
 
> [1.1] Our experiments showed that slice-level CV erroneously boosted the average slice level accuracy on the test set by 30% on Open Access Series of Imaging Studies (OASIS), 29% on Alzheimer’s Disease Neuroimaging Initiative (ADNI), 48% on Parkinson's Progression Markers Initiative (PPMI) and 55% on a local de-novo PD Versilia dataset. (Abstract)
 
> [1.2] Further tests on a randomly labeled OASIS-derived dataset produced about 96% of (erroneous) accuracy (slice-level split) and 50% accuracy (subject-level split), as expected from a randomized experiment. (Abstract)
 
`2.` Yağış E, Atnafu S, Marzi C, et al. Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Scientific Reports.* 2021;11. doi:10.1038/s41598-021-01681-w
 
> [2.1] Our experiments showed that slice-level CV erroneously boosted the average slice level accuracy on the test set by 30% on Open Access Series of Imaging Studies (OASIS), 29% on Alzheimer's Disease Neuroimaging Initiative (ADNI), 48% on Parkinson's Progression Markers Initiative (PPMI) and 55% on a local de-novo PD Versilia dataset. (Abstract)
 
> [2.2] (a) subject-level split, in which all the slices of a subject have been placed either in the training or in the test set, avoiding any form of data leakage; (Methods)
 
> [2.3] We also emphasize the need to document how the CV is implemented, the architecture used, how the different hyperparameter choices/tunings are made and include their values where possible. (Body)
 
> [2.4] Besides, we advocate reproducibility and encourage the community to take a step towards transparency in deep/machine learning in medical image analysis by publicly releasing code, including containers and a link to open datasets59. (Body)
 
`3.` Chhetri L, Kumar A. Optimizing Deep Learning for Brain Tumor Classification: A Comparative Ablation Study of Preprocessing and Augmentation Strategies. *2026 IEEE Guwahati Subsection Conference (GCON).* 2026. doi:10.1109/gcon69192.2026.11649062
 
> [3.1] Recent research indicates that this error causes the accuracy to be inflated by 5-30 percentage points because the model learns patient-specific information such as the skull and ventricle shape rather than tumor information [10], [11]. (Body)
 
> [3.2] Bilateral filtering with augmentation yields 93.86 percent accuracy and 0.991 AUC (Body)
 
> [3.3] Our results show that skull stripping and CLAHE have a detrimental effect on accuracy when combined with augmentation (accuracy of 82 percent), while bilateral filtering with augmentation has an accuracy of 93.86 percent and AUC of 0.991 (p=0.05 compared to augmentation alone). (Abstract)
 
> [3.4] 4) Statistical validation using McNemar's test to confirm that augmentation is the most important factor and that some preprocessing interactions are detrimental II. (Body)
 
> [3.5] The published results are contradictory: skull stripping is beneficial in some papers [14] and detrimental in others [15]. (Body)
 
> [3.6] 1) Patient-level splitting with algorithmic guarantee of zero patient overlap across splits (Body)
 
> [3.7] [10] quantified the accuracy increase of imagelevel splitting over patient-level splitting to be 5-30 percent for several medical imaging tasks. (Body)
 
> [3.8] A controlled patient-level split led to a clinically plausible 87.92 percent baseline, indicating that published accuracies above 98 percent are very likely overstated due to image-level data leakage. (Body)
 
> [3.9] We propose a new "destructive vs. constructive" taxonomy for image preprocessing, which indicates that image preprocessing techniques that preserve anatomical context are more effective than those that remove spatial information. (Abstract)
 
`4.` Firdaus N, Muflikhah L, Perdana RS, Widodo N. Analyzing the Impact of Preprocessing on Deep Learning-based Brain Tumor Classification Using Glioblastoma Magnetic Resonance Image. *Journal of Image and Graphics.* 2026. doi:10.18178/joig.14.4.601-613
 
> [4.1] The results indicate that intensity normalization yields the most stable and discriminative feature representations, achieving the highest accuracy of 89.70% with balanced glioblastoma sensitivity and precision. (Abstract)
 
> [4.2] Conversely, aggressive preprocessing methods distort image intensity distributions and spatial cues, leading to degraded feature learning and reduced performance. (Abstract)
 
`5.` Yonar A. A swarm intelligence-driven hybrid framework for brain tumor classification with enhanced deep features. *Scientific Reports.* 2025;15. doi:10.1038/s41598-025-23820-3
 
> [5.1] Pixel values were normalized to the [0, 1] range prior to training. (Methods)
 
> [5.2] All images were resized to a fixed resolution of 224 × 224 × 3 pixels to ensure compatibility with pretrained CNN architectures. (Methods)
 
> [5.3] To enhance generalization and mitigate overfitting, data augmentation techniques were applied to the training set, including random rotation (±10°), horizontal flipping, shear transformation (0.05), zoom range (±10%), and width/height shift (±10%). (Methods)
 
> [5.4] Yet, deep CNN features are often high-dimensional and redundant, leading to increased computational burden and potential overfitting12–14. (Body)
 
> [5.5] LMNN improved feature separability, while Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO) selected compact subsets. (Abstract)
 
> [5.6] To address these challenges, this study introduces a hybrid framework that integrates CNN-based deep feature extraction, Large Margin Nearest Neighbor (LMNN) metric learning, and swarm intelligence-based feature selection. (Body)
 
> [5.7] The DenseNet201-LMNN-GWO-KNN configuration, termed DenseWolf-K, achieved the best performance with 99.64% accuracy, establishing it as the optimal implementation of the framework. (Abstract)
 
`6.` Iqbal S, Qureshi AN, Alhussein MA, Aurangzeb K, Choudhry IA, Anwar MS. Hybrid deep spatial and statistical feature fusion for accurate MRI brain tumor classification. *Frontiers in Computational Neuroscience.* 2024;18. doi:10.3389/fncom.2024.1423051
 
> [6.1] To be more precise, we scaled each image to a 256 × 256 grid and normalized it to 256 gray levels. (Methods)
 
> [6.2] The recommended hybrid approach achieved a high F1 score of 96.12 ± 0.41, precision of 97.77 ± 0.32, and accuracy of 97.53 ± 0.24, indicating that it has the potential to serve as a valuable tool for pathologists. (Abstract)
 
> [6.3] The radiomics feature fusion with deep learning model produced substantial gains in all parameters with p ≤ 0.05 for the pre-trained CNN architecture and proposed FusionNet model. (Body)
 
`7.` Kang J, Ullah Z, Gwak J. MRI-Based Brain Tumor Classification Using Ensemble of Deep Features and Machine Learning Classifiers. *Sensors (Basel, Switzerland).* 2021;21. doi:10.3390/s21062222
 
> [7.1] Initially, they have used a non-local mean filter to remove the noise from the image. (Body)
 
> [7.2] Experimental results demonstrate that an ensemble of deep features can help improving performance significantly, and in most cases, support vector machine (SVM) with radial basis function (RBF) kernel outperforms other machine learning classifiers, especially for large datasets. (Abstract)
 
`8.` Ahmmed F, Antik AR, Alaly A, Mehnaj S, Huda MSA, Ali MA. BTdiagAI: A Web‐Deployed Hybrid Framework for Brain Tumor Classification Using Optimized MRI Preprocessing and Deep Learning Fusion. *Healthcare Technology Letters.* 2026;13. doi:10.1049/htl2.70053
 
> [8.1] All preprocessing methods achieved greater than 98% accuracy, with CLAHE outperforming others (99.8% on Dataset 1; 99.61% on Dataset 2) while requiring minimal computational resources. (Abstract)
 
`9.` Kondrateva E, Druzhinina P, Dalechina A, et al. Negligible effect of brain MRI data preprocessing for tumor segmentation. *Biomed. Signal Process. Control..* 2022;96:106599. doi:10.1016/j.bspc.2024.106599
 
> [9.1] Finally, we show that the contribution of skull-stripping in data preprocessing is almost negligible if measured in terms of estimated tumor volume. (Abstract)
 
> [9.2] In contrast, inter-subjects anatomy alignment in the form of non-rigid atlas registration is not necessary and intensity equalization steps (denoising, bias-field correction and histogram matching) do not improve models' performance. (Abstract)
 
> [9.3] Second, we propose an explanation for the observed low (or even negative) influence of image intensity correction techniques on model accuracy. (Body)
 
`10.` Khan MSI, Rahman A, Debnath T, et al. Accurate brain tumor detection using deep convolutional neural network. *Computational and Structural Biotechnology Journal.* 2022;20:4733 - 4745. doi:10.1016/j.csbj.2022.08.039
 
> [10.1] For instance, all the MRI images in the Figshare dataset are in.mat type (defined in Matlab). (Methods)
 
> [10.2] After that, we transform all the images into NumPy arrays (available in python) so that our model can take up less space. (Methods)
 
`11.` Safy M, Abd-Ellah M, Bayoumi E, Salama G. Adaptive multi-feature fusion architecture with optimized learning for high-fidelity brain tumor classification in MRI. *Scientific Reports.* 2026;16. doi:10.1038/s41598-026-38672-8
 
> [11.1] Theoretically, applying denoising prior to contrast enhancement risks amplifying residual noise artifacts during the contrast adjustment phase. (Body)
 
> [11.2] The framework achieved 99.05% accuracy, 98.99% recall, 99.52% specificity, 99.08% positive predictive value (PPV), and 99.54% negative predictive value (NPV), consistently surpassed state-of-the-art (SOTA) methods across all reported metrics. (Abstract)
 
> [11.3] These deep representations are combined with handcrafted texture measures based on the Gray-Level Co-occurrence Matrix (GLCM), producing nine unique CNN-GLCM Fused Feature (CGFF) sets. (Abstract)
 
`12.` Alrikabi JM, Muhammad S. Deep Feature Extraction with SVM Classifier for Brain Tumor Classification. *Journal of Education for Pure Science- University of Thi-Qar.* 2024. doi:10.32792/jeps.v14i3.539
 
> [12.1] The outcomes for ResNet-18 with SVM classifier certain that the best deep features were extracted from 'fc1000' and for VggNet-16 with SVMs certain that the best deep features were extracted from 'fc7'. (Conclusion)
 
> [12.2] Specifically, the pretrained CNN with SVM approach produces accuracy ranging from 95.28% to 98.62% across all datasets. (Abstract)
 
`13.` Aurna NF, Yousuf M, Taher KA, Azad A, Moni MA. A classification of MRI brain tumor based on two stage feature level ensemble of deep CNN models. *Computers in biology and medicine.* 2022;146:105539. doi:10.1016/j.compbiomed.2022.105539
 
> [13.1] From the extracted features, most substantial features are selected using Principal Component Analysis (PCA) and fed into the classifier. (Abstract)
 
> [13.2] Through the prominent performance, the proposed model is able to outperform other existing models attaining an average accuracy of 99.13% by optimization of the developed algorithms. (Abstract)
 
`14.` Deepak S, Ameer PM. Automated Categorization of Brain Tumor from MRI Using CNN features and SVM. *Journal of Ambient Intelligence and Humanized Computing.* 2020;12:8357 - 8369. doi:10.1007/s12652-020-02568-w
 
> [14.1] When the amount of available training data is small, the SVM classifier is observed to perform better than the softmax classifier for the CNN features. (Abstract)
 
> [14.2] The proposed model attained an overall classification accuracy of 95.82%, better than the state-of-the-art method. (Abstract)
 
`15.` Deepak S, Ameer PM. Computers in Biology and Medicine.
 
> [15.1] The proposed system records a mean classification accuracy of 98%, outperforming all state-of-the-art methods. (Abstract)
 
`16.` Deep Feature Extraction of Brain Tumor Magnetic Resonance Imaging Classification Based on Hybrid CNN-XGBoost. *International Journal of Intelligent Engineering and Systems.* 2025. doi:10.22266/ijies2025.0331.39
 
> [16.1] Brain tumors can occur in people of all age groups and can originate from various types and sizes of brain tissue.Prompt patient detection is critical for accelerating therapy.Magnetic Resonance Imaging (MRI) is a highly effective technique for detecting chronic disorders like brain tumors.While deep learning techniques dominate medical analytics for health monitoring and brain tumor identification, older machine learning approaches often receive less attention despite their comparable performance.The classification accuracy achieved by these methods is inadequate because they rely solely on texture-based features and employ techniques that fail to capture some key aspects of the images.Additionally, these methods were prone to overfitting.This study substitutes traditional handcrafted feature extraction methods with automated feature extraction using a Convolutional Neural Network (CNN). (Abstract)
 
`17.` Sharif M, Li J, Khan MA, Saleem M. Active deep neural network features selection for segmentation and recognition of brain tumors using MRI images. *Pattern Recognit. Lett..* 2020;129:181-189. doi:10.1016/j.patrec.2019.11.019
 
> [17.1] These features are simply concatenated along with dominant rotated LBP (DRLBP) for better texture analysis. (Abstract)
 
`18.` Hanumanthappa S, Guruprakash CD. Fusion of Handcrafted and Deep-Learning Features for Brain Tumor Detection and Classification Using T1-Weighted Magnetic Resonance Images. *SN Computer Science.* 2024;5. doi:10.1007/s42979-024-03470-4
 
> [18.1] The Support Vector Machine (SVM) along with a linear kernel classifier achieved a mean accuracy of 73.88% from DCA and 93.33% from CCA on the Radiopaedia dataset. (Abstract)
 
`19.` Chhetri L, Datta A, Ghosal P. Attention-Enhanced Swin Transformers for Robust Brain Tumor Classification Under Patient-Level Data Splitting. *2026 IEEE Guwahati Subsection Conference (GCON).* 2026. doi:10.1109/gcon69192.2026.11648823
 
> [19.1] Evaluation of five architectures reveals degradations reaching 3.71 percent under patient-level splitting, validating widespread data leakage. (Abstract)
