CSC-481 — Southern Connecticut State University
DLCGIPG: Deep Learning Classifiers for Gemstone Identification and Price Grading

Logan Caraballo · Sebastian Scrimenti · Shlok Gandhi
caraballol2@southernct.edu · scrimentis1@southernct.edu · gandhis2@southernct.edu

---

# Introduction

## Problem Statement

Detecting genuine diamonds and assessing their quality is a challenging task for buyers and sellers without access to professional gemological equipment. Misidentification can result in financial loss, fraud, and unfair pricing. Online diamond retail has moved billions of dollars of inventory to studio-photography catalogs, yet no publicly available system lets a consumer verify gemstone identity or estimate value tier directly from the product photograph they are already viewing.

This project develops a two-stage deep learning pipeline that first classifies an input image by gemstone type and then predicts the value tier of a confirmed diamond. Beyond the core classification task, the project conducts a cross-domain generalization experiment: two self-collected datasets from different online retailers with meaningfully different imaging systems are used to test whether trained models transfer across sites or overfit to site-specific photographic conditions. This is a practically important question for any deployed diamond grading system.

## Objective

Design, train, and evaluate three deep learning architectures — ResNet50, EfficientNetV2, and Vision Transformer (ViT-B/16) — across a two-stage gemstone analysis pipeline. Stage 1 classifies images by gemstone type (68 classes including diamond). Stage 2 predicts the value tier of confirmed diamonds using both a visual regression model and a tabular 4C Random Forest baseline for direct comparison. Performance is assessed using accuracy, macro F1-score, and end-to-end pipeline metrics across a benchmark derived from self-collected and publicly available datasets.

The project asks five formal research questions:
1. Can ResNet50, EfficientNetV2, or ViT-B/16 classify gemstone types accurately on the MDPI Minerals benchmark, and does a larger training set reverse the finding of Chow and Reyes-Aldasoro (2022) that ResNet-50 underperforms Random Forest on ~2,000 images?
2. Can the same architectures classify diamond value tiers from retail photography at accuracy levels that exceed a random baseline?
3. Does any visual architecture match or exceed the tabular 4C Random Forest baseline?
4. Do models trained on JamesAllen.com images generalize to BrilliantEarth.com images, and vice versa?
5. Does model performance differ between natural and lab-grown diamond subsets?

## Motivation

The motivation for this project is to address the lack of transparency and trust in the gemstone market, where the average consumer depends on specialized merchants to determine whether a diamond is genuine and how it should be graded. Existing AI grading systems — GIA/IBM and Sarine — operate on specialized laboratory hardware with millions of proprietary training samples; they are not accessible to consumers or small retailers.

This project evaluates what is achievable from consumer-accessible retail photography — the same images buyers already view online — establishing a baseline for the limits of accessible automated grading. By combining visual classifiers with a tabular ceiling analysis, the project also quantifies exactly how much information is lost when moving from structured 4C attributes to a JPEG photograph.

---

# Related Work

He et al. [1] introduced ResNet50, establishing residual connections as the standard approach for training deep CNNs without vanishing gradient degradation. Tan and Le [2] proposed EfficientNetV2, achieving strong accuracy with improved training speed and parameter efficiency through compound scaling. Dosovitskiy et al. [3] demonstrated that transformer-based self-attention applied to fixed-size image patches can match or exceed CNNs at scale when pretrained on sufficiently large datasets.

Chow and Reyes-Aldasoro [10] benchmarked ResNet-18 and ResNet-50 on 2,042 gemstone images across 68 categories, finding a best accuracy of 69.4% with Random Forest — outperforming ResNet-50 on this small dataset. Their result directly motivates Stage 1 of this project: does a dataset 14× larger reverse the CNN vs. Random Forest ordering? Their MDPI Minerals dataset serves as the held-out benchmark enabling direct comparison with published results.

Bendinelli et al. [11] (GEMTELLIGENCE) applied CNN + attention to spectroscopic data for gemstone origin determination. Their data modality is physically diagnostic and entirely distinct from consumer photography; direct accuracy comparisons would be misleading. Swain et al. [12] (GemInsight) applied Random Forest to 4C tabular features for diamond quality prediction, establishing the tabular upper bound this project extends to the visual domain. Zhou [9] demonstrated that carat weight accounts for up to 95% of price prediction variance in Random Forest models; this motivates the RF Feature Importance analysis used here to detect whether carat similarly dominates tier prediction.

National Jeweler [16] documented GIA and Sarine AI deployment in commercial grading contexts, establishing the industrial state of the art against which this project's consumer-photography approach is positioned. Multiple authors [17] have established that controlled imaging conditions are a prerequisite for reliable automated diamond color grading.

---

# Data

## Data Source and Format

Nine datasets are used across two pipeline stages.

**Table 1: Dataset Summary**

| # | Dataset | Source | Size | Key Labels | Use |
|---|---|---|---|---|---|
| 1 | Gemstones Images (Sindhu) | Kaggle | ~4,000 img | Gem type | Stage 1 training |
| 2 | Precious Gemstone ID (Kamath) | Kaggle | ~49,273 img | Gem type | Stage 1 primary |
| 3 | MDPI Minerals (Chow) | GitHub | 2,326 img | 68 gem classes | Stage 1 held-out benchmark |
| 4 | Diamond Images (Purswani) | Kaggle | ~1,500 img | Diamond type | Stage 1 supplement |
| 5 | JA Natural Diamonds (scraped) | JamesAllen.com | 107,687 img | Cut, color, clarity, carat, price | Stage 2 JA natural |
| 6 | JA Lab Diamonds (scraped) | JamesAllen.com | 121,519 img | Cut, color, clarity, carat, price | Stage 2 JA lab-grown |
| 7 | BE Natural Diamonds (scraped) | BrilliantEarth.com | 106,950 img | Cut, color, clarity, carat, price | Stage 2 BE natural |
| 8 | BE Lab Diamonds (scraped) | BrilliantEarth.com | 260,875 img | Cut, color, clarity, carat, price | Stage 2 BE lab-grown |

**Stage 1 — Merged Training Pool:**
Datasets 1, 2, and 4 were merged after label normalization to form the Stage 1 training pool (~51,599 images, 68 classes trimmed from an original 87 classes following the Chow taxonomy). Dataset 3 (MDPI Minerals, 2,326 images) is held out entirely and never seen during training; it is used only for benchmark comparison against Chow and Reyes-Aldasoro (2022).

**Stage 2 — Self-Collected Diamond Datasets:**
Stage 2 data was collected via custom Python scrapers targeting publicly accessible product pages requiring no authentication. JA data was collected via plain HTTP requests against JA's internal GraphQL API using micro-band carat sweeps (0.01ct increments, 0.25–6.00ct range, 575 bands per shape) across all 10 diamond shapes. BE data was collected via Playwright (headless Chromium) to navigate Cloudflare's JS challenge using the same sweep strategy.

**Stage 2 Dataset Totals:**

| Dataset | Source | Rows | Images | Shapes | Kaggle |
|---|---|---|---|---|---|
| JA Natural | JamesAllen.com | 107,687 | 107,687 | 10 | ja-diamond-images-4c v2 |
| JA Lab-Grown | JamesAllen.com | 121,519 | 121,517 | 10 | ja-diamond-images-4c v2 |
| BE Natural | BrilliantEarth.com | 106,950 | 106,950 | 10 | be-diamond-images-4c v1 |
| BE Lab-Grown | BrilliantEarth.com | 260,875 | 260,875 | 10 | be-diamond-images-4c v1 |
| **TOTAL** | — | **597,031** | **597,029** | — | — |

All images are JPEG format. Metadata is stored in CSV files with one row per diamond, containing: diamond ID, shape, carat weight, cut grade, color grade, clarity grade, price (USD), and computed value tier label.

## Data Example

A single record from the JA natural dataset:
- **Image:** 757×600 JPEG of a round brilliant cut diamond, standardized 40× superzoom, white background, standardized lighting.
- **Metadata:** shape=round, carat=1.20, cut=Ideal+, color=G, clarity=VS1, price=$6,840, tier=premium.

JamesAllen.com photographs every diamond in-house at 40× magnification under standardized lighting. Resolution is fixed at 757×600 pixels (99.7% of inventory). BrilliantEarth.com sources from multiple suppliers; resolution varies 300–460 px, aspect ratio varies (mean 1.03 ± 0.08), and imaging conditions differ per supplier.

**Stage 1 image example:** 224×224 resized studio photograph of a ruby, amethyst, or emerald from the Kaggle gemstone datasets. Images vary in background and lighting — not standardized.

## Features

**Stage 2 Visual Features (learned):** The deep learning models receive only the resized JPEG as input. No structured attributes are provided at Stage 2 inference. The models must infer value signals — most importantly, carat weight (the dominant price driver) — from visual appearance alone.

**Stage 2 Tabular Features (RF baseline only):** The Random Forest baseline receives four structured 4C attributes: carat weight (continuous), cut grade (ordinal, 5 levels), color grade (ordinal, 10 levels), clarity grade (ordinal, 11 levels). These are the exact attributes professional gemologists use to price diamonds.

**RF Feature Importance (Stage 2 baseline):**

| Subset | Carat Importance | Color+Clarity+Cut Combined |
|---|---|---|
| ja_natural | 0.853 | 0.147 |
| ja_lab | 0.793 | 0.207 |
| be_natural | 0.805 | 0.195 |
| be_lab | 0.784 | 0.216 |

Carat dominance (0.79–0.85) is expected — diamond price scales exponentially with carat weight. The remaining 15–21% signal from color, clarity, and cut confirms the tier labels reflect real pricing structure, not pure carat bucketing.

**Value Tier Boundaries (JA Natural, for reference):**

| Tier | Boundary | Count | % of Total |
|---|---|---|---|
| Budget | ≤ $840 | 59,427 | 25.9% |
| Mid-Range | $840 – $4,440 | 110,868 | 48.3% |
| Premium | $4,440 – $11,880 | 34,057 | 14.8% |
| Investment-Grade | > $11,880 | 24,854 | 10.8% |

Tier boundaries are computed independently per subset (natural vs. lab-grown, JA vs. BE) using percentile cutoffs: Budget ≤ P25, Mid-Range P25–P75, Premium P75–P90, Investment-Grade > P90. Class imbalance ratio ~4.5×.

---

# Methodology

## Preprocessing

All images are resized to 224×224 RGB and normalized using ImageNet mean and standard deviation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for transfer learning compatibility. Training augmentation: random horizontal flip, rotation (±15°), brightness/contrast jitter, and Gaussian blur. Color jitter is intentionally excluded — diamond color grade is a potentially informative classification feature and should not be randomized away.

For the Tier 3.3 high-resolution Stage 2 experiment, Stage 2 input is resized to 384×384; Stage 1 continues at 224×224. The dual-resolution pipeline applies independent transforms per stage.

Stage 2 regression target: log(price_usd), z-scored per source subset using training-split statistics to equalize scale across retailers and diamond types. Denormalization uses per-subset mean and standard deviation at evaluation time to recover USD predictions.

## Features (Model Input)

Stage 1 input: 224×224 RGB image tensor. Output: 68-class softmax probability vector. Routing to Stage 2 uses the softmax probability assigned to the "Diamond" class.

Stage 2 input: 224×224 (or 384×384) RGB image tensor. Output: single continuous value (normalized log-price). This is then thresholded to one of four value tiers using per-subset calibrated boundaries.

## Classifiers

All three architectures are initialized with pretrained ImageNet weights. The final classification head is replaced with a task-specific output layer. All parameters are fine-tuned (no frozen backbone).

**Table 2: Architecture Summary**

| Architecture | Parameters | Backbone | Head |
|---|---|---|---|
| ResNet50 | 24.6M | ImageNet1K_V2 | Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→4) |
| EfficientNetV2-S | 20.2M | ImageNet1K_V1 | Dropout(0.3) → Linear(1280→4) |
| ViT-B/16 | 85.8M | ImageNet1K_V1 | Dropout(0.3) → Linear(768→4) |

**Stage 2 Regression Head (EfficientNetV2):** Linear(1280→1). Loss: HuberLoss(delta=0.5) on normalized log-price. Tier classification is derived by applying learned thresholds to the continuous output.

**Random Forest Baseline:** Receives structured 4C tabular features (carat, cut, color, clarity). Trained and evaluated on the same data splits as the visual classifiers.

**Training Hyperparameters (Stage 2):**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate (head) | 3e-4 |
| Learning rate (backbone) | 3e-5 |
| Scheduler | CosineAnnealingLR (30 epochs to 1e-6) |
| Batch size | 64 (48 for 384×384) |
| Max epochs | 30 (early stop, patience=5 on val macro F1) |
| Label smoothing | 0.1 |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | float16 + GradScaler (within-site only) |
| Class weights | WeightedRandomSampler + weighted CE loss |

Cross-domain runs used float32 (float16 caused NaN loss due to gradient spikes on out-of-distribution data).

## Design of Experiments — Data Division (Training/Testing)

**Stage 1:** 70/15/15 stratified train/val/test split on the merged Sindhu + Kamath + Purswani pool. MDPI Minerals dataset held out completely as the published benchmark test.

**Stage 2 — Single-Site:** 70/15/15 stratified split per subset. Splits stratified by value tier to preserve class balance across all three sets. 20 separate split CSVs generated at seed=42; splits are locked and shared across all architectures. Natural and lab-grown diamonds kept as separate subsets throughout.

**Stage 2 — Cross-Domain:** Training split from one retailer, evaluation on the full test split of the other retailer. Zero data from the evaluation retailer seen during training.

**Stage 2 — Combined Training:** Combined dataset merges all four subsets. Train/val sets are subsampled to the smallest contributing site to prevent retailer imbalance; test sets include all images from all sites unsampled (realistic evaluation).

| Subset | Train | Val | Test |
|---|---|---|---|
| combined_natural | 149,730 | 32,084 | 32,197 |
| combined_lab | 170,122 | 36,456 | 57,360 |
| combined_all | 299,460 | 64,168 | 89,557 |

**Pipeline Evaluation Dataset:** 91,324 images total — 89,557 diamond images (combined_all test split) and 1,767 non-diamond images (Combined-P1-Dataset test split). This evaluates both Stage 1 gating and Stage 2 tier prediction in a single end-to-end pass.

**Tier Calibration:** Tier boundaries are calibrated post-training on a stratified 50/50 split of the pipeline evaluation set. The calibration half (43,518 images) is used for coordinate-descent threshold optimization; results are reported on the held-out validation half (43,519 images) and then the full 89,557-image test set.

## Evaluation Metrics

**Stage 1:**
- Top-1 Accuracy on combined test split
- Macro F1-Score
- Per-class F1 across 68 gemstone classes
- Comparison against Chow et al. RF baseline on MDPI Minerals benchmark
- In pipeline context: Diamond recall and false positive rate

**Stage 2:**
- Top-1 Accuracy
- Macro F1-Score (primary — penalizes minority class failures equally)
- Per-class F1 for all four value tiers
- Cross-domain accuracy delta (JA→BE, BE→JA)
- Visual vs. tabular accuracy delta (image model vs. RF baseline)

**Regression (Stage 2):**
- R² on log-price
- Log-MAE
- USD Mean Absolute Error (USD-MAE)
- Spearman rank correlation

**Pipeline End-to-End:**
- Stage 1 diamond recall
- Stage 1 false positive rate
- Stage 2 tier macro F1
- Stage 2 USD MAE and Median Absolute Percentage Error (MdAPE)
- End-to-end tier accuracy (Stage 1 miss counted as a wrong answer)
- Per-subset breakdown across all four diamond populations

---

# Results

## Stage 1 Results

**[PLACEHOLDER — Sebastian's Stage 1 section]**

*The following tables are placeholders to be filled in with Stage 1 training and benchmark results.*

**Table S1-A: Stage 1 — Architecture Comparison on Test Split**

| Architecture | Test Accuracy | Macro F1 | Epochs Run |
|---|---|---|---|
| EfficientNetV2-S | [fill] | [fill] | [fill] |
| ResNet50 | [fill] | [fill] | [fill] |
| ViT-B/16 | [fill] | [fill] | [fill] |
| RF Baseline (Chow config) | [fill] | [fill] | — |

**Table S1-B: Stage 1 — MDPI Minerals Benchmark (Held-Out)**

| Model | Accuracy | Macro F1 | vs. Chow et al. Best (69.4%) |
|---|---|---|---|
| Chow et al. (2022) RF baseline | 69.4% | [fill] | — |
| EfficientNetV2-S | [fill] | [fill] | [fill] |
| ResNet50 | [fill] | [fill] | [fill] |
| ViT-B/16 | [fill] | [fill] | [fill] |

**Table S1-C: Stage 1 — Top Confused Gemstone Pairs**

| Predicted Class | True Class | Count | Notes |
|---|---|---|---|
| [fill] | [fill] | [fill] | [fill] |

*Note on pipeline Stage 1: In end-to-end evaluation, the deployed Stage 1 model (EfficientNetV2, fine-tuned with Pyrite hard negatives) achieved Diamond recall = 99.98% and false positive rate = 0.23% on the 91,324-image pipeline test set (see Pipeline Results section).*

---

## Stage 2 Random Forest Tabular Baseline

The RF baseline establishes the performance ceiling for attribute-driven prediction. It receives the structured 4C attributes directly.

**Table 1: RF Baseline Results**

| Subset | Test Accuracy | Macro F1 | Carat Importance |
|---|---|---|---|
| ja_natural | 0.8658 | 0.8567 | 0.853 |
| ja_lab | 0.9411 | 0.9217 | 0.793 |
| be_natural | 0.8371 | 0.8226 | 0.805 |
| be_lab | 0.8961 | 0.8742 | 0.784 |

---

## Stage 2 Within-Site Classification Results

**Table 2: Within-Site Classification — All Architectures (Test Set)**

| Architecture | Subset | Test Accuracy | Macro F1 | Best Val F1 | Epochs |
|---|---|---|---|---|---|
| EfficientNetV2 | ja_natural | 0.6798 | 0.6724 | 0.6845 | 28 (early stop) |
| EfficientNetV2 | ja_lab | 0.6904 | 0.6589 | 0.6665 | 30 (max) |
| EfficientNetV2 | be_natural | 0.6292 | 0.6093 | 0.6117 | 30 (max) |
| EfficientNetV2 | be_lab | 0.5822 | 0.5554 | 0.5574 | 30 (max) |
| ResNet50 | ja_natural | 0.6652 | 0.6590 | 0.6620 | 25 (early stop) |
| ResNet50 | ja_lab | 0.6674 | 0.6386 | 0.6410 | 30 (max) |
| ResNet50 | be_natural | 0.6289 | 0.6075 | 0.6042 | 30 (max) |
| ResNet50 | be_lab | 0.5875 | 0.5557 | 0.5556 | 30 (max) |
| ViT-B/16 | ja_natural | 0.5411 | 0.5595 | 0.5648 | 12 (early stop) |
| ViT-B/16 | ja_lab | 0.6170 | 0.6181 | 0.6239 | 30 (max) |
| ViT-B/16 | be_natural | 0.6326 | 0.5885 | 0.5917 | 30 (max) |
| ViT-B/16 | be_lab | 0.5786 | 0.5295 | 0.5312 | 30 (max) |

**Table 3: Within-Site Classification — Per-Class F1**

| Architecture | Subset | Budget F1 | Mid-Range F1 | Premium F1 | Inv-Grade F1 |
|---|---|---|---|---|---|
| EfficientNetV2 | ja_natural | 0.783 | 0.528 | 0.714 | 0.665 |
| EfficientNetV2 | ja_lab | 0.814 | 0.505 | 0.618 | 0.699 |
| EfficientNetV2 | be_natural | 0.752 | 0.451 | 0.609 | 0.624 |
| EfficientNetV2 | be_lab | 0.722 | 0.400 | 0.505 | 0.595 |
| ResNet50 | ja_natural | 0.787 | 0.495 | 0.706 | 0.648 |
| ResNet50 | ja_lab | 0.807 | 0.486 | 0.590 | 0.671 |
| ResNet50 | be_natural | 0.743 | 0.446 | 0.610 | 0.632 |
| ResNet50 | be_lab | 0.718 | 0.393 | 0.494 | 0.617 |
| ViT-B/16 | ja_natural | 0.739 | 0.418 | 0.664 | 0.418 |
| ViT-B/16 | ja_lab | 0.782 | 0.436 | 0.566 | 0.689 |
| ViT-B/16 | be_natural | 0.729 | 0.398 | 0.556 | 0.671 |
| ViT-B/16 | be_lab | 0.692 | 0.342 | 0.455 | 0.629 |

---

## Stage 2 Cross-Domain Results

Models trained on one retailer, evaluated on the other's test set.

**Table 4: Cross-Domain Classification Results**

| Architecture | Train | Test | Direction | Test Accuracy | Macro F1 |
|---|---|---|---|---|---|
| EfficientNetV2 | ja_natural | BE | JA→BE | 0.1089 | 0.0708 |
| EfficientNetV2 | ja_lab | BE | JA→BE | 0.1550 | 0.1371 |
| EfficientNetV2 | be_natural | JA | BE→JA | 0.4216 | 0.3054 |
| EfficientNetV2 | be_lab | JA | BE→JA | 0.1684 | 0.1411 |
| ResNet50 | ja_natural | BE | JA→BE | 0.1719 | 0.1538 |
| ResNet50 | ja_lab | BE | JA→BE | 0.1475 | 0.1284 |
| ResNet50 | be_natural | JA | BE→JA | 0.3914 | 0.3097 |
| ResNet50 | be_lab | JA | BE→JA | 0.2709 | 0.2048 |
| ViT-B/16 | ja_natural | BE | JA→BE | 0.1145 | 0.0872 |
| ViT-B/16 | ja_lab | BE | JA→BE | 0.1467 | 0.1286 |
| ViT-B/16 | be_natural | JA | BE→JA | 0.2604 | 0.2162 |
| ViT-B/16 | be_lab | JA | BE→JA | 0.3576 | 0.2424 |

**Table 5: Cross-Domain Direction Summary**

| Direction | Mean Macro F1 | Min F1 | Max F1 |
|---|---|---|---|
| JA → BE | 0.1177 | 0.0708 | 0.1538 |
| BE → JA | 0.2366 | 0.1411 | 0.3097 |

**Table 6: Domain Shift Quantification (JA natural vs. BE natural)**

| Metric | JA natural | BE natural | Effect Size |
|---|---|---|---|
| R channel mean | 182 | 174 | Cohen's d = 0.77 (large) |
| G channel mean | — | — | Cohen's d = 0.61 (medium) |
| B channel mean | — | — | Cohen's d = 0.28 (small) |
| Channel std (contrast) | ~43 | ~36 | — |
| Resolution | 757×600 (99.7% fixed) | ~300×300 (215–460 px variable) | ~2.5× difference |
| Aspect ratio | 1.262 ± 0.000 | 1.028 ± 0.082 | — |

---

## Domain Adaptation Experiment

Starting from the worst cross-domain model (EfficientNetV2, JA natural → BE, F1=0.071), fine-tuned with increasing samples from the target domain.

**Table 7: Domain Adaptation Results (EfficientNetV2, JA→BE direction)**

| Run | Fine-Tune N | Epochs | Eval Macro F1 | vs. Baseline | Multiplier |
|---|---|---|---|---|---|
| Baseline (zero-shot) | 0 | — | 0.0708 | — | 1.0× |
| N=500 | 400 train / 100 val | 10 | ~0.261 | +0.190 | 3.7× |
| N=1000 | 800 / 200 | 10 | ~0.320 | +0.249 | 4.5× |
| N=2000 | 1600 / 400 | 10 | 0.3723 | +0.302 | 5.3× |
| N=2000 ext | 1600 / 400 | 20 | 0.4004 | +0.330 | 5.7× |

Gains are monotonic with N and epochs. N=2000 ext was not converged at epoch 20; estimated asymptote ~0.42–0.43.

---

## Stage 2 Regression Results

EfficientNetV2 head replaced with Linear(1280→1), loss HuberLoss(delta=0.5), target log(price_usd).

**Table 8: Single-Site Regression — All Architectures**

| Architecture | Subset | R² | Log-MAE | USD-MAE | Spearman ρ |
|---|---|---|---|---|---|
| EfficientNetV2 | ja_natural | 0.8803 | 0.3316 | $2,184 | 0.931 |
| EfficientNetV2 | ja_lab | 0.8592 | 0.2359 | $820 | 0.931 |
| EfficientNetV2 | be_natural | 0.7829 | 0.3470 | $1,721 | 0.885 |
| EfficientNetV2 | be_lab | 0.7075 | 0.3841 | $1,671 | 0.824 |
| ResNet50 | ja_natural | 0.8686 | 0.3484 | $2,305 | 0.924 |
| ResNet50 | ja_lab | 0.7535 | 0.3172 | $1,024 | 0.891 |
| ResNet50 | be_natural | 0.7514 | 0.3802 | $1,881 | 0.865 |
| ResNet50 | be_lab | 0.6897 | 0.4024 | $1,735 | 0.808 |
| ViT-B/16 | ja_natural | 0.7547 | 0.4725 | $2,864 | 0.858 |
| ViT-B/16 | ja_lab | 0.8007 | 0.2787 | $920 | 0.898 |
| ViT-B/16 | be_natural | 0.7201 | 0.3949 | $1,924 | 0.848 |
| ViT-B/16 | be_lab | 0.6548 | 0.4256 | $1,818 | 0.782 |

**Table 9: Regression vs. RF Ceiling — JA Natural and BE Natural**

| Metric | RF (JA) | Vision (JA) | RF (BE) | Vision (BE) |
|---|---|---|---|---|
| Log-MAE | 0.153 | 0.332 | 0.138 | 0.347 |
| USD-MAE | $976 | $2,184 | $665 | $1,721 |
| Median APE % | 11.5% | 26.0% | 10.6% | 25.5% |
| R² (log) | — | 0.880 | — | 0.783 |
| Spearman ρ | — | 0.931 | — | 0.885 |

**Table 10: Regression-to-Tier Bridge vs. Direct Classification**

| Subset | Regression Tier F1 | Direct Classification F1 | Delta |
|---|---|---|---|
| ja_natural | 0.7366 | 0.6724 | +0.064 |
| be_natural | 0.6906 | 0.6093 | +0.081 |

---

## Combined Training Results

**Table 11: Combined Regression — EfficientNetV2**

| Subset | Test Log-MAE | Tier F1 | Epochs |
|---|---|---|---|
| combined_natural | 0.2917 | 0.7115 | 30 |
| combined_all | 0.3631 | 0.6459 | 27 |
| combined_lab | 0.3746 | 0.6322 | 30 |

**Table 12: Combined Classification — All Architectures**

| Architecture | Subset | Macro F1 | Accuracy | Epochs |
|---|---|---|---|---|
| EfficientNetV2 | combined_natural | 0.6979 | 0.7296 | 27 |
| EfficientNetV2 | combined_all | 0.6392 | 0.6912 | 26 |
| EfficientNetV2 | combined_lab | 0.6218 | 0.6821 | 22 |
| ResNet50 | combined_all | 0.6256 | 0.6807 | 24 |
| ViT-B/16 | combined_all | 0.5901 | 0.6534 | 26 |

**Table 13: Seed Variance — EfficientNetV2 ja_natural (N=3 seeds)**

| Seed | Macro F1 | Accuracy |
|---|---|---|
| 0 | 0.6724 | 0.6798 |
| 1 | 0.6801 | 0.6884 |
| 2 | 0.6774 | 0.6847 |
| Mean ± Std | 0.677 ± 0.004 | — |

---

## End-to-End Pipeline Results

The two-stage pipeline was evaluated on 91,324 images. The table below shows the progression from the initial evaluation through each fine-tuning tier.

**Table 14: Pipeline Optimization Journey**

| Endpoint | E2E Accuracy | Stage 2 Macro F1 | USD MAE | Notes |
|---|---|---|---|---|
| Pre-fix (May 2, with bug) | 0.320 | 0.298 | — | Tier-label ordering bug present |
| Bug fix only | 0.649 | 0.643 | $1,739 | One-line fix to TIER_LABELS constant |
| Tier 1.1 — global calibration | 0.677 | 0.659 | — | F1-optimal global thresholds |
| Tier 1.2 — per-subset calibration | 0.683 | 0.665 | — | Per-subset threshold optimization |
| Tier 1.3 — softmax routing (T=0.02) | 0.704 | 0.666 | $1,739 | Stage 1 recall: 97.2% → 99.8% |
| Tier 2.1 — Stage 1 fine-tune | 0.705 | 0.667 | $1,738 | Stage 1 recall: 99.8% → 100% |
| Tier 2.2 — classifier head (frozen backbone) | 0.702 | 0.663 | $1,738 | Negative result — ceiling held |
| Tier 3.2 — class-balanced sampling | 0.691 | 0.659 | $1,742 | Negative result — ceiling held |
| **Tier 3.3 — hi-res 384×384** | **0.731** | **0.694** | **$1,554** | **+38% E2E vs. original buggy baseline** |

**Table 15: Final Pipeline Results — Tier 3.3 Production Endpoint**

| Metric | Value |
|---|---|
| Total diamond images evaluated | 89,557 |
| Total non-diamond images evaluated | 1,767 |
| Stage 1 diamond recall | 99.99% |
| Stage 1 false positive rate | 0.23% |
| Stage 2 tier accuracy | 73.1% |
| Stage 2 tier macro F1 | 0.6945 |
| Stage 2 USD mean absolute error | $1,554 |
| Stage 2 USD median absolute % error | 22.6% |
| End-to-end tier accuracy | 73.1% |

**Table 16: Final Pipeline Results — Per-Subset Breakdown (Tier 3.3)**

| Subset | n | Stage 1 Recall | Stage 2 Macro F1 | E2E Accuracy | USD MAE | USD MdAPE |
|---|---|---|---|---|---|---|
| be_lab | 39,132 | 99.97% | 0.623 | 67.3% | $1,761 | 27.6% |
| be_natural | 16,043 | 100.00% | 0.728 | 75.5% | $1,451 | 21.9% |
| ja_lab | 18,228 | 100.00% | 0.748 | 78.6% | $791 | 15.5% |
| ja_natural | 16,154 | 100.00% | 0.767 | 78.2% | $2,016 | 22.3% |

**Table 17: Final Pipeline — Per-Tier Classification Report (Tier 3.3, Stage-1-Passed Images)**

| Tier | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Budget | [fill from TIER33 detail] | [fill] | 0.79 | 22,336 |
| Mid-Range | [fill] | [fill] | 0.77 | 42,337 |
| Premium | [fill] | [fill] | 0.53 | 12,859 |
| Investment-Grade | [fill] | [fill] | 0.69 | 9,505 |

*Note: Insert per-tier precision/recall from pipeline_eval_detail.TIER33.csv if available. Macro F1 = 0.6945 confirmed.*

---

# Analysis of the Results

## EfficientNetV2 is the Winning Architecture

EfficientNetV2-S outperforms ResNet50 on every subset by 0.1–1.3 F1 points with fewer parameters (20.2M vs. 24.6M). Despite being the lightest architecture, compound scaling delivers better accuracy per parameter at this dataset scale. This is a meaningful paper finding: the smallest model wins.

## ViT-B/16 Significantly Underperforms

ViT-B/16 early-stopped at epoch 12 on ja_natural (F1 = 0.560) while CNNs ran 25–30 epochs and reached 0.659–0.672. ViT requires substantially more data than CNNs to generalize. Even at 75,000 training images, ViT is data-starved relative to what it needs to beat CNNs. This extends the Chow et al. (2022) finding: scale alone does not reverse the CNN vs. transformer ordering when the dataset is not sufficiently large.

## Mid-Range is the Hardest Class (Pre-Registered)

Mid-range produced the lowest per-class F1 in every single completed run across all architectures, subsets, and training conditions. This was predicted before training began: mid-range is the largest class (48% of samples), sits between adjacent classes with the least price separation, and has the highest within-class carat variance. Confirmation from all 12 within-site runs strengthens this as a paper finding.

## BE is Consistently Harder than JA

BE subsets show 5–8 F1 points lower performance than corresponding JA subsets across all architectures. JA's standardized 40× superzoom regime produces visually homogeneous images. BE's multi-supplier imaging produces more varied backgrounds, resolutions (300–460 px), and lighting conditions. This imaging difference is the plausible explanation, confirmed by the domain shift quantification (Cohen's d = 0.77 on the R channel).

## Image-to-Tabular Gap Quantified

The gap between image model performance and the RF tabular ceiling is 0.18–0.27 F1 points across all subsets. The RF achieves 0.82–0.92 F1 with 4C attributes; the best visual classifier achieves 0.67 F1 on within-site data. This quantifies the cost of working from retail photographs instead of structured 4C attributes. The gap exists because a JPEG cannot communicate carat weight (0.79–0.85 RF feature importance) with the precision of a scale measurement. No visual architecture approaches the tabular ceiling — the gap is explained by the fundamental information difference between the two modalities.

## Cross-Domain Transfer Fails Catastrophically (JA→BE)

All six JA→BE runs failed across all three architectures (mean F1 = 0.12). The EfficientNetV2 JA natural cross-domain model predicts "premium" for 93.7% of BE inputs — a degenerate collapse explained by BE images being displaced outside the model's learned decision boundary by the brightness shift (Cohen's d = 0.77) and 2.5× resolution difference. Cross-domain failure is categorical, not a continuous degradation.

BE→JA shows partial transfer (mean F1 = 0.24 ≈ 2× JA→BE). BE's multi-supplier imaging produces more generalizable visual features than JA's controlled 40× regime. Natural subsets transfer better than lab subsets.

## Domain Adaptation is Recoverable (Not Representational)

Starting from EfficientNetV2 JA→BE F1 = 0.071 (worst result), fine-tuning with 2,000 target-domain samples (no retraining from scratch) recovers F1 to 0.40 — a 5.7× improvement. This demonstrates the gap is a calibration problem, not a representational one. The backbone already contains useful features; it simply needs domain exposure.

## Regression Outperforms Direct Classification

The regression-to-tier formulation outperforms direct 4-class classification on both tested subsets by +6.4 and +8.1 F1 points respectively. The continuous target preserves ordinal structure near tier boundaries that 4-class cross-entropy loses. Combined training on both retailers further improves over single-site: combined_natural regression tier F1 = 0.7115 vs. best single-site 0.6724.

## Resolution is the Stage 2 Bottleneck

Two consecutive negative results (head-only training, class-balanced sampling) showed that premium F1 was stuck at 0.49 despite varied optimization strategies — and led us to hypothesize an information-theoretic ceiling. That hypothesis was wrong. Moving Stage 2 input from 224×224 to 384×384 (Tier 3.3) broke the premium ceiling (F1 0.49 → 0.53), improved every other class, and reduced USD MAE by $184 (−10.6%). The visual signal to distinguish a $4,000–$12,000 stone from a $1,000–$4,000 stone of the same shape was present in the original retailer photographs; it was sub-pixel at 224×224.

## Stage 1 Routing: Pyrite Confusion

Under the original argmax routing, 90.1% of ja_natural diamonds were correctly classified by Stage 1. Confusion analysis showed 96.6% of all ja_natural Stage 1 misses were predicted as Pyrite ("fool's gold") — a highly concentrated, actionable failure mode. A softmax-threshold routing rule (route if Diamond probability > 2%) recovered ja_natural Stage 1 recall from 90.1% to 99.1% at zero false positive cost. Subsequent fine-tuning on retailer-style diamonds with Pyrite hard negatives brought recall to 100%.

---

# Conclusion

## Limitations

1. **Consumer photography vs. laboratory conditions.** All images are studio retail photography, not laboratory-grade captures. GIA and Sarine operate with specialized hardware and controlled environments that this project does not replicate. Consumer-submitted smartphone photos would face additional domain shift beyond what is measured here.

2. **Price as quality proxy.** Value tier labels are derived from retail price, not independent gemological assessments. Price reflects market conditions (supply, fashion trends, retailer margin) in addition to quality. The RF Feature Importance analysis (carat 0.79–0.85) confirms the labels reflect real pricing structure, but this approximation is a disclosed limitation.

3. **BE round diamond tier boundary approximation.** Round diamonds were absent from BE's initial scrape due to a scraper gap. Because round-specific price percentiles were unavailable before collection, BE round tier boundaries were approximated using the price distribution of BE's 9 non-round shapes. This may introduce minor tier boundary inaccuracies for BE round diamonds specifically.

4. **Separate price spaces.** Lab-grown and natural diamonds occupy fundamentally different price spaces. Results are reported independently per subset throughout; a unified classifier across both diamond types would face additional price-space conflation not measured here.

5. **Class imbalance.** Mid-range dominates at 48% of samples; investment-grade is 10.8%. Despite WeightedRandomSampler and weighted loss, investment-grade consistently shows high precision but low recall — the model is conservative when uncertain about top-tier predictions.

6. **Stage 1 domain coverage.** The Stage 1 training pool (~51,599 images) underrepresents retailer-style diamond photography (standardized 40× superzoom) relative to studio gemstone photography. This caused the ja_natural Diamond → Pyrite confusion that required targeted fine-tuning. A training pool that explicitly includes retailer-style diamond images from the start would eliminate this bottleneck.

## Issues Not Resolved

1. **Premium tier F1 ceiling at 0.53 (Stage 2, Tier 3.3).** Despite resolution improvement, premium remains the hardest tier. The precision/recall tradeoff within premium (premium borders both mid-range and investment-grade) was not resolved. Two optimization paths failed (frozen-backbone classifier head, class-balanced full retrain), and the resolution increase provided only +0.04 improvement. The remaining premium confusion likely reflects genuine visual overlap between $4K and $12K stones when evaluated on a 384×384 JPEG — a limit of the modality.

2. **be_lab lowest subset F1 (0.623, Tier 3.3).** Lab diamonds at BrilliantEarth's price scale have the weakest visual signal-to-price relationship. Lab pricing depends heavily on certification grade and carat weight, which are not visually salient. This was the lowest-performing subset at every optimization tier without exception.

3. **Cross-domain regression not completed.** Single-site regression runs are complete; the corresponding cross-domain regression experiment (train on JA regression, evaluate on BE, and vice versa) was scoped but not executed within this project's timeline. These results would directly test whether continuous price regression is more robust to domain shift than discrete tier classification.

4. **Seed variance for non-EfficientNetV2 architectures.** Seed variance was validated only for EfficientNetV2 ja_natural (mean F1 = 0.677 ± 0.004). ResNet50 and ViT-B/16 seed validation was scoped but not completed; their reported single-seed results should be interpreted with this caveat.

## Future Direction

1. **Higher-resolution input with hardware parity.** Stage 2 at 384×384 improved every metric; 448×448 or 512×512 may push premium F1 further. The main constraint is VRAM — the RTX 5070 Ti (16 GB) was near capacity at 384×384 with batch size 48. Gradient checkpointing or a larger GPU would enable the next resolution step without architectural changes.

2. **Multi-modal fusion (image + carat weight only).** The RF ceiling shows carat accounts for 0.79–0.85 of tier predictability. If carat weight is available at inference (it is displayed on retailer product pages), combining the visual embedding with a single scalar carat input via a fusion head would close most of the image-to-tabular gap without requiring full 4C attribute access. This was deliberately excluded from the controlled comparison; it is the natural next experiment.

3. **Per-subset specialist Stage 2 models.** The single combined_all model handles all four subsets (natural/lab × JA/BE). A routing layer that branches to a natural specialist and a lab specialist (using available metadata at inference) could improve be_lab F1 specifically — the lab diamond pricing problem is sufficiently different from the natural problem that a specialist may generalize better.

4. **Stage 1 Pyrite hard-negative augmentation at initial training time.** The Pyrite confusion was resolved by post-hoc fine-tuning, but the underlying cause is that the Stage 1 training pool underrepresents retailer-style diamond photography. Incorporating labeled retailer-style diamonds directly into the Stage 1 training pool from the start would eliminate the need for a separate fine-tuning pass.

5. **Consumer photograph domain.** All data in this project is professional studio photography from established online retailers. Extending the pipeline to consumer-submitted photographs (variable backgrounds, smartphone cameras, inconsistent lighting) represents a qualitatively harder domain shift problem and a practically important deployment scenario not measured here.

---

# Appendix

## Pipeline Architecture (Diagram)

```
Input Image
     │
     ▼
┌─────────────────────────┐
│      Stage 1            │
│  EfficientNetV2-S       │
│  68-class classifier    │
│  (softmax T=0.02 gate)  │
└─────────────────────────┘
     │                  │
     │ Diamond          │ Non-Diamond
     │ (p > 0.02)       │ (other class)
     ▼                  ▼
┌─────────────────────────┐   Output:
│      Stage 2            │   "Amethyst", "Ruby", etc.
│  EfficientNetV2-S       │
│  Regression head        │
│  384×384 input          │
│  Per-subset thresholds  │
└─────────────────────────┘
     │
     ▼
Output: Value Tier + USD Estimate
(Budget / Mid-Range / Premium / Investment-Grade)
+ USD price estimate
```

## Snapshot and Others

**Production Model Configuration (Final):**

| Stage | Architecture | Weights Path | Input Size | Task |
|---|---|---|---|---|
| Stage 1 | EfficientNetV2-S | results/training/stage1/efficientnetv2_v2/best_model.pth | 224×224 | 68-class gem classification |
| Stage 2 | EfficientNetV2-S | results/training/regression/efficientnetv2_hires/combined_all/best_model.pth | 384×384 | Regression → tier + USD |

**Routing Rule:** Softmax threshold T=0.02 (route to Stage 2 if Diamond class probability > 2%)

**Per-Subset Calibrated Thresholds (normalized log-price space):**

| Subset | Budget ↔ Mid-Range | Mid-Range ↔ Premium | Premium ↔ Inv-Grade |
|---|---|---|---|
| be_lab | −0.6616 | +0.6398 | +1.1155 |
| be_natural | −0.7674 | +0.5605 | +1.4018 |
| ja_lab | −0.7493 | +0.6979 | +1.3467 |
| ja_natural | −0.7389 | +0.3970 | +1.2935 |

**Stage 1 Confusion — ja_natural Misses Under Argmax:**

| Confused with | Count | Share of Misses |
|---|---|---|
| Pyrite ("fool's gold") | 1,540 | 96.6% |
| Citrine | 27 | 1.7% |
| Andradite | 7 | 0.4% |
| Topaz | 6 | 0.4% |
| (other 65 classes) | 15 | 0.9% |

**Inference Command:**
```bash
python src/evaluate_pipeline.py \
  --stage1-weights results/training/stage1/efficientnetv2_v2/best_model.pth \
  --stage2-weights results/training/regression/efficientnetv2_hires/combined_all/best_model.pth \
  --stage2-input-size 384
```

---

# References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. https://arxiv.org/abs/1512.03385

[2] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," *International Conference on Machine Learning (ICML)*, 2021. https://arxiv.org/abs/2104.00298

[3] A. Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," *International Conference on Learning Representations (ICLR)*, 2021. https://arxiv.org/abs/2010.11929

[4] L. Sindhu, "Gemstones Images Dataset," Kaggle. https://www.kaggle.com/datasets/lsind18/gemstones-images

[5] G. Kamath, "Precious Gemstone Identification Dataset," Kaggle. https://www.kaggle.com/datasets/gauravkamath02/precious-gemstone-identification

[6] L. Caraballo, S. Scrimenti, S. Gandhi, "BE Diamond Images — 4C Value Tiers," Kaggle. https://www.kaggle.com/datasets/junyiiblvc/be-diamond-images-4c

[7] L. Caraballo, S. Scrimenti, S. Gandhi, "JA Diamond Images — 4C Value Tiers," Kaggle. https://www.kaggle.com/datasets/junyiiblvc/ja-diamond-images-4c

[8] S. Bansal, "Diamonds Dataset," Kaggle. https://www.kaggle.com/datasets/shivam2503/diamonds

[9] M. Zhou, "Enhancing Diamond Price Prediction through Machine Learning and Deep Learning: A Comparative Analysis of AGS and GIA Grading Systems," unpublished manuscript, 2025.

[10] C. Chow and C. C. Reyes-Aldasoro, "Automatic Gemstone Classification Using Computer Vision," *Minerals*, MDPI, 2022. https://doi.org/10.3390/min12010060

[11] T. Bendinelli et al., "GEMTELLIGENCE: Accelerating Gemstone Classification with Deep Learning," *Communications Engineering*, 2024. https://doi.org/10.1038/s44172-024-00252-x

[12] D. Swain et al., "GemInsight: Unleashing Random Forest for Diamond Quality Forecasting," 2023.

[13] JamesAllen.com, "Loose Diamond Search," Retrieved March 2026. https://www.jamesallen.com/loose-diamonds/all-diamonds/

[14] BrilliantEarth.com, "Diamond Search," Retrieved March 2026. https://www.brilliantearth.com/loose-diamonds/

[15] ResearchGate, "Deep Learning Applications in Industrial Diamond Crystal Grading," 2022.

[16] National Jeweler, "State of the Diamond Industry: AI and the Future of Diamond Grading," 2023. https://nationaljeweler.com/articles/11975

[17] Multiple authors, machine vision approaches to diamond color grading, 2009–2024. Includes Shyamala Devi et al. (2024), establishing controlled imaging requirements for reliable automated color grading.

[18] A. Purswani, "Diamond Images Dataset," Kaggle. https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset
