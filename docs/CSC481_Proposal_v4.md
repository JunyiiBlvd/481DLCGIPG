# Using Deep Learning Classifiers for Gemstone Identification and Diamond Value Tier Classification

## CSC-481 — Project Proposal (Revised v4)
### Southern Connecticut State University

| Name | Sections | Email |
|---|---|---|
| Sebastian | 3, 4 | scrimentis1@southernct.edu |
| Logan | 5, 6 | caraballol2@southernct.edu |
| Shlok | 1, 2 | gandhis2@southernct.edu |

---

## Abstract

This project systematically compares three deep learning architectures — ResNet50, EfficientNetV2, and Vision Transformer (ViT-B/16) — across a two-stage gemstone analysis pipeline. Stage 1 classifies input images by gemstone type (diamond vs. ruby, emerald, sapphire, and others). Stage 2 classifies confirmed diamonds into four value tiers derived from price-percentile boundaries applied to structured 4C attributes. Two large-scale Stage 2 datasets were self-collected via custom scrapers: JamesAllen.com (229,206 labeled diamonds, 229,204 images) and BrilliantEarth.com (367,825 labeled diamonds, 367,825 images), totaling 597,029 labeled records across 10 diamond shapes, both natural and lab-grown. The primary novel contribution of Stage 2 is a cross-domain generalization experiment — models trained on one retailer's dataset are evaluated on the other's — testing whether visual classifiers learn transferable diamond quality features or site-specific imaging artifacts. A Random Forest classifier trained on structured 4C tabular features serves as the performance ceiling for attribute-driven prediction. No published study has applied all three architectures at this scale on a self-collected diamond dataset, directly compared visual classifiers against a tabular 4C baseline, or conducted a cross-site domain generalization experiment across matched diamond retailer datasets.

---

## 1. Introduction

### 1.1 Problem Statement

Detecting real diamonds and grading their quality is a challenging task, especially for buyers and sellers who lack access to professional gemological equipment. Misidentification can lead to financial loss, fraud, and unfair pricing. This project develops a two-stage machine learning pipeline that first classifies input images by gemstone type, and then predicts the value tier of confirmed diamonds using a combination of visual features and structured quality attributes.

Beyond the core classification task, the project introduces a cross-domain generalization experiment: two self-collected datasets from different online retailers with meaningfully different imaging systems are used to test whether trained models transfer across sites or overfit to site-specific photographic conditions. This is a practically important question for any deployed diamond grading system.

### 1.2 Objective

Design, train, and evaluate three deep learning architectures — ResNet50, EfficientNetV2, and Vision Transformer (ViT-B/16) — across a two-stage gemstone analysis system. Stage 1 classifies images by gemstone type. Stage 2 classifies confirmed diamonds into value tiers, using both visual classifiers and a tabular 4C Random Forest baseline for direct comparison. Model performance is assessed using accuracy, F1-score, inference latency, and cross-domain accuracy delta across a unified benchmark derived from self-collected and publicly available datasets.

### 1.3 Motivation

The motivation for this project is to address the lack of transparency and trust in the gemstone market, where the average consumer depends on specialized merchants to determine whether a diamond is genuine and how it should be graded. This reliance invites misinformation, unfair pricing, and fraud, as buyers may have no way to verify grading accuracy. By developing a machine learning model that can identify real diamonds from images and assign an estimated value tier using their visual appearance and structured attributes, this project aims to make diamond assessment more accessible, objective, and consistent.

Existing AI grading systems — GIA/IBM, Sarine — operate on specialized laboratory hardware with millions of proprietary training samples. This project evaluates what is achievable from consumer-accessible retail photography, the same images buyers already view online, establishing a baseline for the limits of accessible automated grading.

### 1.4 Research Questions

1. Can ResNet50, EfficientNetV2, or ViT-B/16 classify gemstone types accurately on the MDPI Minerals benchmark, and does a larger training set reverse the finding of Chow and Reyes-Aldasoro [10] that ResNet-50 underperforms Random Forest on ~2,000 images?
2. Can the same architectures classify diamond value tiers from retail photography at accuracy levels that exceed a random baseline?
3. Does any visual architecture match or exceed the tabular 4C Random Forest baseline, or does the information gap between image and structured attributes prevent visual classifiers from reaching tabular performance levels?
4. Do models trained on JamesAllen.com images generalize to BrilliantEarth.com images, and vice versa? What cross-domain accuracy delta is observed?
5. Does model performance differ between natural and lab-grown diamond subsets?

### 1.5 Novel Contribution

No published study has: (1) applied ResNet50, EfficientNetV2, and Vision Transformer in a controlled three-way comparison on a self-collected labeled diamond image dataset of this scale (597,029 images); (2) directly compared visual classifier performance against a tabular 4C baseline on the same task and same data split; or (3) conducted a cross-site cross-domain generalization experiment comparing two major online diamond retailers with different imaging systems. This project addresses all three gaps simultaneously.

### 1.6 Related Work

He et al. [1] introduced ResNet50, establishing residual connections as the standard approach for training deep CNNs without vanishing gradient degradation. Tan and Le [2] proposed EfficientNetV2, achieving strong accuracy with improved training speed and parameter efficiency through compound scaling. Dosovitskiy et al. [3] demonstrated that transformer-based self-attention applied to fixed-size image patches can match or exceed CNNs at scale when pretrained on sufficiently large datasets.

Chow and Reyes-Aldasoro [10] benchmarked ResNet-18 and ResNet-50 on 2,042 gemstone images across 68 categories, finding a best accuracy of 69.4% with Random Forest — outperforming ResNet-50 on this small dataset. Their result motivates this project's Stage 1 design: does a dataset 14× larger reverse the CNN vs. Random Forest ordering? Bendinelli et al. [11] (GEMTELLIGENCE) applied CNN + attention to spectroscopic data (UV-Vis, FTIR, ICP-MS) for gemstone origin determination. Their data modality is physically diagnostic and entirely distinct from consumer photography; direct accuracy comparisons would be misleading. Swain et al. [12] (GemInsight) applied Random Forest to 4C tabular features for diamond quality prediction, establishing the tabular upper bound this project extends to the visual domain. Zhou [9] demonstrated that carat weight accounts for up to 95% of price prediction variance in Random Forest models; this motivates the RF Feature Importance analysis used here to detect whether carat similarly dominates tier prediction, which would indicate a labeling artifact.

Researchgate [15] applied ResNet-34 and U-Net to industrial diamond crystal grading from camera images, demonstrating that controlled imaging conditions are achievable outside laboratory settings. National Jeweler [16] documented GIA and Sarine AI deployment in commercial grading contexts, establishing the industrial state of the art against which this project's consumer-photography approach is positioned. Multiple authors [17] have established that controlled imaging conditions are a prerequisite for reliable automated diamond color grading; JA's standardized 40× superzoom regime partially satisfies these conditions.

### 1.7 Assumptions and Scope

This project assumes that standardized retail photography contains learnable visual signal correlated with diamond value tiers, and that price-percentile-derived tier labels are a reasonable proxy for quality tiers. Both assumptions are tested rather than taken on faith: the tabular RF baseline establishes whether 4C attributes predict tiers, and model accuracy relative to random baseline establishes whether images carry learnable signal.

Images are studio photography, not consumer smartphone photos. A consumer-facing deployment would face additional domain shift beyond what is measured here; this is acknowledged as future work. Lab-grown and natural diamonds occupy fundamentally different price spaces and are kept as separate subsets throughout; all results are reported independently for each subset. Price reflects market conditions in addition to quality; tier boundaries are an approximation of quality tiers, mitigated by 4C attribute availability for every record.

---

## 2. Data

### 2.1 Dataset Overview

Nine datasets are used across two pipeline stages. Table 1 summarizes their roles.

**Table 1: Dataset Summary**

| # | Dataset | Source | Size | Key Labels | Use |
|---|---|---|---|---|---|
| 1 | Gemstones Images (Sindhu) | Kaggle | ~4,000 img | Gem type (multi-class) | Stage 1 training |
| 2 | Precious Gemstone ID (Kamath) | Kaggle | ~2,000 img | Gem type (multi-class) | Stage 1 supplement |
| 3 | MDPI Minerals (Chow) | GitHub/hybchow | 2,326 img | Gem type (68 classes) | Stage 1 published benchmark |
| 4 | Diamonds Dataset (Bansal) | Kaggle | 54K records | 4C attrs + price (tabular) | RF baseline supplement (tabular) |
| 5 | Diamond Images (Purswani) | Kaggle | ~1,500 img | Diamond type label | Stage 1 supplement |
| 6 | JA Natural Diamonds (scraped) | JamesAllen.com | 107,687 img + 4C | Cut, color, clarity, carat, price | Stage 2 primary — JA natural |
| 7 | JA Lab Diamonds (scraped) | JamesAllen.com | 121,519 img + 4C | Cut, color, clarity, carat, price | Stage 2 primary — JA lab-grown |
| 8 | BE Natural Diamonds (scraped) | BrilliantEarth.com | 106,950 img + 4C | Cut, color, clarity, carat, price | Stage 2 cross-domain — BE natural |
| 9 | BE Lab Diamonds (scraped) | BrilliantEarth.com | 260,875 img + 4C | Cut, color, clarity, carat, price | Stage 2 cross-domain — BE lab-grown |

### 2.2 Stage 1 Data: Gemstone Type Classification

Datasets 1, 2, and 5 are merged after label reconciliation to form the Stage 1 training pool (~7,500 images). Dataset 3 — the MDPI Minerals dataset (Chow and Reyes-Aldasoro [10]), 2,326 images across 68 gemstone classes — is used as a held-out benchmark enabling direct comparison with published results. A traditional ML baseline replicating [10]'s best configuration — Random Forest with RGB 8-bin color histogram and LBP texture features — is also implemented, placing all three deep learning architectures against the published state of the art on this exact dataset. This positions Stage 1 results within the existing literature rather than evaluating architectures only against each other.

### 2.3 Stage 2 Data: Value Tier Classification

#### 2.3.1 Data Collection

Stage 2 training data was collected from two major online diamond retailers via custom Python scrapers targeting each site's internal search API. All data is from publicly accessible product pages requiring no authentication. Scrapers identified themselves as academic research tools via User-Agent string.

**James Allen (JA):** Scraped via JA's internal GraphQL API (`service-api/ja-product-api/diamond/v/2/`) using micro-band carat sweeps (0.01ct increments, 0.25–6.00ct range, 575 bands per shape) across all 10 diamond shapes, for both natural and lab-grown diamonds. Transport: plain `requests` (no browser required). Rate: 0.7 req/sec with ±20% randomized jitter. Metadata collection: ~3 hours; image download: ~8 hours (AMD Ryzen 9 9900X, RTX 5070 Ti). Checkpoint/resume implemented via per-shape JSON checkpoint files keyed by carat band. JA photographs every diamond in-house at 40× magnification under standardized lighting, white balance, and camera conditions — the most controlled consumer-accessible diamond photography available. This imaging consistency partially satisfies the controlled imaging requirements identified in [17] for reliable automated color grading.

**Brilliant Earth (BE):** Scraped via BE's internal search endpoint (`/api/product/getSearchResults/`), accessed through Playwright (headless Chromium) — plain HTTP was blocked by Cloudflare's JS challenge. Same micro-band carat sweep strategy. Fancy-colored lab diamonds (non-D–Z color scale) were excluded at the API query level to maintain comparability with the JA dataset, which carries only colorless and near-colorless stones; excluded stones were logged. BE diamonds are sourced from multiple suppliers with varied imaging setups, making BE's imaging domain meaningfully different from JA's proprietary 40× superzoom system. This imaging domain difference is intentional — it is the basis of the cross-domain generalization experiment.

#### 2.3.2 Final Dataset State

**James Allen:**

| Metric | Value |
|---|---|
| Total labeled rows | 229,206 |
| Natural diamonds | 107,687 |
| Lab-grown diamonds | 121,519 |
| Shapes | 10 (all normalized) |
| Images on disk | 229,204 |
| Missing images | 2 (CDN dead — confirmed permanent) |
| Kaggle dataset | junyiiblvc/ja-diamond-images-4c (private, v2) |

**JA Tier Boundaries (computed independently per subset):**

| Tier | Natural Boundary | Lab-Grown Boundary | Total Count |
|---|---|---|---|
| Budget (≤P25) | ≤ $840 | ≤ $960 | 59,427 |
| Mid-Range (P25–P75) | $840 – $4,440 | $960 – $4,120 | 110,868 |
| Premium (P75–P90) | $4,440 – $11,880 | $4,120 – $6,050 | 34,057 |
| Investment-Grade (>P90) | > $11,880 | > $6,050 | 24,854 |

**JA Shape Distribution (post-normalization):**

| Shape | Count |
|---|---|
| Round | 57,256 |
| Oval | 33,754 |
| Emerald | 26,109 |
| Pear | 24,147 |
| Princess | 21,690 |
| Cushion | 18,657 |
| Radiant | 16,880 |
| Heart | 11,175 |
| Marquise | 10,750 |
| Asscher | 8,788 |

> Note: JA API returns "Cushion Modified" and "Square Radiant" as shape names. These were normalized to "cushion" and "radiant" respectively (16,771 and 408 rows) to match BE's vocabulary and enable cross-domain training on matched shape categories.

**Brilliant Earth:**

| Metric | Value |
|---|---|
| Total labeled rows | 367,825 |
| Natural diamonds | 106,950 |
| Lab-grown diamonds | 260,875 |
| Shapes | 10 |
| Images on disk | 367,825 |
| Missing images | 0 (rows with unresolvable CDN failures dropped from final CSV) |
| Kaggle dataset | junyiiblvc/be-diamond-images-4c (private, v2) |

**BE Tier Distribution:**

| Tier | Count |
|---|---|
| Budget | 95,534 |
| Mid-Range | 179,059 |
| Premium | 53,676 |
| Investment-Grade | 39,556 |

> Exact BE tier boundaries (computed independently from BE's own price distribution — not derived from JA thresholds) are stored in `be_scraper/output/be_tier_stats.json`.

**Combined Totals:**

| Metric | James Allen | Brilliant Earth | Combined |
|---|---|---|---|
| Labeled rows | 229,206 | 367,825 | **597,031** |
| Images on disk | 229,204 | 367,825 | **597,029** |
| Natural | 107,687 | 106,950 | 214,637 |
| Lab-grown | 121,519 | 260,875 | 382,394 |
| Missing images | 2 (0.00%) | 0 (0.00%) | 2 |
| Shapes | 10 | 10 | 10 (matched vocabulary) |

#### 2.3.3 Value Tier Labels

Four value tier labels are derived from the price distribution of each dataset using percentile boundaries: Budget (≤25th percentile), Mid-Range (25th–75th), Premium (75th–90th), and Investment-Grade (>90th). Boundaries are computed **independently per subset** — natural and lab-grown diamonds are labeled using their own price distributions, and JA and BE boundaries are computed separately from each site's own data. Class balance ratio approximately 0.20; weighted random sampling and weighted cross-entropy loss are used during training to address imbalance. Macro F1 is reported alongside accuracy to ensure minority class performance is visible.

#### 2.3.4 Stage 2 Tabular Baseline

A Random Forest classifier trained on structured 4C features (cut, color, clarity, carat) from the scraped datasets serves as the performance ceiling for attribute-driven prediction, against which all three visual classifiers are directly compared on the same task and same data splits. RF Feature Importance analysis tests whether carat dominates tier prediction, as it dominates price prediction [9] — which would indicate a labeling artifact and will be disclosed in the final report.

#### 2.3.5 Data Split

70/15/15 stratified split (train/val/test) applied consistently across all classifiers. Natural and lab-grown diamonds kept as separate subsets throughout. For cross-domain experiments: training split from one site, evaluation split from the other (no data from the evaluation site seen during training).

### 2.4 Preprocessing

All images resized to 224×224 RGB, normalized using ImageNet mean/std for transfer learning compatibility. Augmentation applied to training data: random horizontal flip, rotation (±15°), brightness/contrast jitter, Gaussian blur. Color jitter is intentionally excluded to preserve diamond color as a potentially informative classification feature. 70/15/15 stratified split applied consistently across all classifiers and both stages.

---

## 3. Methodology

### 3.1 Architectures

Three deep learning architectures are evaluated across both stages:

- **ResNet50 [1]:** Canonical residual CNN with skip connections enabling training of very deep networks without vanishing gradients. Well-characterized on medium-scale datasets and serves as the performance anchor for the architecture comparison. As noted by Chow and Reyes-Aldasoro [10], ResNet-50 underperformed Random Forest on ~2,000 gemstone images — this project tests whether a 14× larger Stage 1 training set and a substantially larger Stage 2 dataset reverse that result.

- **EfficientNetV2 [2]:** Compound-scaled CNN with improved training speed and parameter efficiency. State of the art in efficient convolutional design; hypothesized to perform best in the moderate-dataset regime relative to model capacity.

- **Vision Transformer (ViT-B/16) [3]:** Pretrained on ImageNet-21k via the `timm` library. Fine-tuned with DeiT-style training: label smoothing (0.1), CutMix augmentation, and stochastic depth, compensating for the data requirements of attention-based architectures relative to CNNs.

All three architectures are initialized with pretrained ImageNet weights and trained on identical data splits with the same preprocessing pipeline, ensuring a fair comparison.

### 3.2 Stage 1: Gemstone Type Identification

Multi-class classifier distinguishing diamond from ruby, emerald, sapphire, amethyst, and other gemstone categories, using the merged Stage 1 training pool (~7,500 images from Datasets 1, 2, and 5). All three architectures are trained and evaluated on the same 70/15/15 split. Performance is additionally benchmarked against Dataset 3 (MDPI Minerals, held-out, 2,326 images across 68 classes) to enable direct comparison with Chow and Reyes-Aldasoro [10]. The Stage 1 Random Forest baseline replicates [10]'s best configuration: RGB 8-bin color histogram + LBP texture features.

If a gemstone image is classified as non-diamond at Stage 1, that label becomes the final pipeline output. Only images classified as diamond proceed to Stage 2.

### 3.3 Stage 2: Diamond Value Tier Classification

For confirmed diamonds, each of the three architectures predicts one of four value tiers: Budget, Mid-Range, Premium, or Investment-Grade. Four experimental conditions are evaluated:

1. **Within-site (JA→JA):** Train on JA training split, evaluate on JA test split.
2. **Within-site (BE→BE):** Train on BE training split, evaluate on BE test split.
3. **Cross-domain (JA→BE):** Train on JA training split, evaluate on BE test split (no BE data seen during training).
4. **Cross-domain (BE→JA):** Train on BE training split, evaluate on JA test split (no JA data seen during training).

All four conditions are run independently for natural and lab-grown subsets. The Random Forest tabular baseline is trained on structured 4C attributes and evaluated on matched test splits from both sites for direct visual-vs-tabular comparison.

---

## 4. Design of Experiments

### 4.1 Stage 1 Experiment Setup

Input images are passed through a multi-class classifier to identify gemstone type. All three architectures (ResNet50, EfficientNetV2, ViT-B/16) are trained and tested on the same dataset split using the same preprocessing pipeline. Performance is compared against the Stage 1 Random Forest baseline and benchmarked against the MDPI Minerals published results [10].

### 4.2 Stage 2 Experiment Setup

Images confirmed as diamonds at Stage 1 (or ground-truth diamond labels for isolated Stage 2 evaluation) are passed to the value tier classifier. All four experimental conditions (within-site and cross-domain) are evaluated for all three architectures. The tabular RF baseline is evaluated on the same test splits for direct visual-vs-tabular comparison.

### 4.3 Primary Metrics

| Metric | Stage | Rationale |
|---|---|---|
| Top-1 Accuracy | 1, 2 | Standard classification benchmark |
| Macro F1-Score | 1, 2 | Handles class imbalance; weights all classes equally |
| Weighted F1-Score | 2 | Appropriate for ordinal value tier classes |
| Cross-Domain Accuracy Delta (JA→BE, BE→JA) | 2 | Primary novel metric — quantifies generalization across imaging domains |
| Visual vs. Tabular Accuracy Delta | 2 | Core comparison: image classifier vs. 4C RF baseline on same task |

### 4.4 Secondary Metrics

| Metric | Stage | Rationale |
|---|---|---|
| Confusion Matrix | 1, 2 | Systematic inter-class confusion patterns |
| Mean Absolute Grade Error (MAGE) | 2 | Ordinal distance of tier mispredictions |
| Inference Latency (ms/image) | 1, 2 | Deployment feasibility |
| GPU Training Time (min/epoch) | 1, 2 | Compute cost for reproducibility |
| RF Feature Importance | 2 baseline | Tests whether carat dominates tier prediction — labeling artifact check |
| Natural vs. Lab-Grown Accuracy Delta | 2 | Tests whether architecture performance differs by diamond origin |

---

## 5. Dataset Provenance and Limitations

### 5.1 Stage 2 Dataset Provenance

Stage 2 data was collected via custom Python scrapers targeting publicly accessible product pages requiring no authentication. JA data was collected via plain HTTP requests against JA's internal GraphQL API. BE data was collected via Playwright (headless Chromium) to navigate Cloudflare's JS challenge. Both scrapers operated within observable rate limits with randomized jitter and identified themselves as academic research tools. Metadata collection: JA ~3 hours, BE longer due to Cloudflare overhead. Image download: JA ~8 hours. Hardware: AMD Ryzen 9 9900X, RTX 5070 Ti, single machine.

### 5.2 Image Quality and Consistency

JamesAllen.com photographs every diamond in-house at 40× magnification under standardized lighting, white balance, and camera conditions. This standardized imaging regime is a significant advantage over arbitrary stock photography: domain shift between JA training data and JA inference is minimized for controlled-imaging deployment scenarios. BrilliantEarth.com sources diamonds from multiple suppliers with varied imaging setups, making BE images less homogeneous across the inventory. This difference is intentional: JA's controlled imaging may produce models more brittle when transferred to BE's varied imaging, or conversely, BE's variety may produce more robust cross-site generalizers. The cross-domain experiment tests this directly.

### 5.3 Remaining Limitations

- **Consumer photography vs. laboratory conditions.** Images are retail photography, not laboratory-grade captures. GIA and Sarine [16] operate with specialized hardware and controlled environments this project does not replicate. Consumer deployment would face additional domain shift.
- **Separate price spaces.** Lab-grown and natural diamonds occupy fundamentally different price spaces. Treated as separate subsets throughout; results reported independently.
- **Price as quality proxy.** Tier boundaries derived from price are an approximation of quality tiers. RF Feature Importance analysis [Section 4.4] will test whether carat dominates tier prediction, indicating a labeling artifact to be disclosed.
- **Class imbalance.** Class balance ratio approximately 0.20. Addressed via weighted random sampling and weighted cross-entropy loss. Macro F1 reported alongside accuracy.
- **BE round diamond tier boundaries.** Round diamonds were absent from BE's initial scrape (a scraper gap discovered during EDA) and were appended via a supplemental download. Because round-specific price percentiles were unavailable before collection, BE round tier boundaries were approximated using the price distribution of BE's 9 non-round shapes. This approximation may introduce minor tier boundary inaccuracies for BE round diamonds specifically; the exact boundaries used are stored in `be_scraper/output/be_tier_stats.json`.
- **BE missing images.** 107 BE rows with permanently unresolvable CDN failures (HTTP 403 or <3 KB placeholder) were dropped from the final labeled CSV. All 367,825 remaining records have verified images on disk.

---

## 6. Timeline

| Weeks | Tasks | Status |
|---|---|---|
| 1–2 | Dataset acquisition and exploration, literature review, environment setup | ✅ Complete |
| 3–6 | JA scraper development, data collection, labeling pipeline | ✅ Complete |
| 6–9 | BE scraper development, cross-site data collection, Kaggle upload | ✅ Complete (March 29, 2026) |
| 9–10 | Preprocessing pipeline, Stage 1 training (all three architectures) | 🔜 In progress |
| 10–11 | Stage 1 evaluation, Stage 2 within-site training (JA and BE) | 🔜 Next |
| 11–12 | Stage 2 cross-domain experiments (JA→BE, BE→JA) + RF baseline | 🔜 Scheduled |
| 12–13 | Full evaluation, metric computation, error analysis | 🔜 Scheduled |
| 13–15 | Report writing, visualization, presentation preparation | 🔜 Scheduled |

---

## 7. Team Responsibilities

| Member | Sections | Primary Responsibilities |
|---|---|---|
| Shlok | 1, 2 | Problem framing, literature review, data collection pipeline (JA and BE scrapers), dataset labeling pipeline, EDA, preprocessing and splits |
| Sebastian | 3, 4 | Architecture selection and implementation (ResNet50, EfficientNetV2, ViT-B/16), training pipeline, experimental setup, evaluation code |
| Logan | 5, 6 | Dataset provenance documentation, limitations analysis, timeline management, cross-domain experiment design |

All members contribute to final report writing and presentation preparation.

---

## 8. References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. https://arxiv.org/abs/1512.03385

[2] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," *International Conference on Machine Learning (ICML)*, 2021. https://arxiv.org/abs/2104.00298

[3] A. Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," *International Conference on Learning Representations (ICLR)*, 2021. https://arxiv.org/abs/2010.11929

[4] L. Sindhu, "Gemstones Images Dataset," Kaggle, Retrieved March 2026. https://www.kaggle.com/datasets/lsind18/gemstones-images

[5] G. Kamath, "Precious Gemstone Identification Dataset," Kaggle, Retrieved March 2026. https://www.kaggle.com/datasets/gauravkamath02/precious-gemstone-identification

[6] S. Bansal, "Diamonds Dataset," Kaggle, Retrieved March 2026. https://www.kaggle.com/datasets/shivam2503/diamonds

[7] A. Purswani, "Diamond Images Dataset," Kaggle, Retrieved March 2026. https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset

[8] H. Lakhani, "Natural Diamonds Prices + Images," Kaggle, Retrieved March 2026. https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images

[9] M. Zhou, "Enhancing Diamond Price Prediction through Machine Learning and Deep Learning: A Comparative Analysis of AGS and GIA Grading Systems," unpublished manuscript, 2025.

[10] C. Chow and C. C. Reyes-Aldasoro, "Automatic Gemstone Classification Using Computer Vision," *Minerals*, MDPI, 2022. https://doi.org/10.3390/min12010060

[11] T. Bendinelli et al., "GEMTELLIGENCE: Accelerating Gemstone Classification with Deep Learning," *Communications Engineering*, 2024. https://doi.org/10.1038/s44172-024-00252-x

[12] D. Swain et al., "GemInsight: Unleashing Random Forest for Diamond Quality Forecasting," August 2023.

[13] JamesAllen.com, "Loose Diamond Search," Retrieved March 2026. https://www.jamesallen.com/loose-diamonds/all-diamonds/

[14] BrilliantEarth.com, "Diamond Search," Retrieved March 2026. https://www.brilliantearth.com/loose-diamonds/

[15] ResearchGate, "Deep Learning Applications in Industrial Diamond Crystal Grading," 2022. ResNet-34/U-Net architecture for diamond crystal classification from camera images.

[16] National Jeweler, "State of the Diamond Industry: AI and the Future of Diamond Grading," 2023. https://nationaljeweler.com/articles/11975

[17] Multiple authors, diamond color grading via machine vision, 2009–2024. Includes Shyamala Devi et al. (2024); survey of machine vision approaches establishing controlled imaging requirements for reliable automated color grading.

[18] Gemini CLI (Google DeepMind), used for parallel scraper execution and dataset validation, March 2026.
