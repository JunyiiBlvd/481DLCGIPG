# DLCGIPG — Deep Learning Classifiers for Gemstone Identification and Price Grading

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> CSC-481 Project — Southern Connecticut State University

A two-stage deep learning pipeline for gemstone analysis. Stage 1 classifies gem species across 68 classes. Stage 2 predicts diamond price tiers and USD price estimates using visual features.

---

## Overview

Three architectures are compared across both stages:

- ResNet-50 (He et al., 2016)
- EfficientNetV2-S (Tan & Le, 2021)
- Vision Transformer ViT-B/16 (Dosovitskiy et al., 2021)

Stage 2 experiments cover within-site, cross-domain, combined-retailer, and regression variants. EfficientNetV2-S is the strongest performer across all tasks.

---

## Project Structure

```
DLCGIPG/
├── src/
│   ├── models.py                    # Stage 2 model factory (4-class head)
│   ├── train.py                     # Single-site classification training
│   ├── train_regression.py          # Single-site regression training
│   ├── train_combined.py            # Combined-retailer classification training
│   ├── train_regression_combined.py # Combined-retailer regression training
│   ├── train_stage1.py              # Stage 1 gem species classifier training
│   ├── diamond_dataset.py           # PyTorch Dataset for diamond images
│   ├── evaluate_pipeline.py         # End-to-end pipeline evaluation
│   ├── rf_baseline.py               # Random Forest tabular baseline
│   ├── evaluate.py                  # Model evaluation utilities
│   └── aggregate_results.py         # Results aggregation
├── scripts/
│   ├── build_combined_splits.py     # Generate combined train/val/test CSVs
│   ├── build_combined_dataset.py    # Generate combined pipeline manifest
│   ├── run_within_site.sh
│   ├── run_cross_domain.sh
│   ├── run_regression.sh
│   ├── run_combined_classification.sh
│   └── run_combined_regression.sh
├── ja_scraper/                      # JamesAllen.com scraper
├── be_scraper/                      # BrilliantEarth.com scraper
├── data/
│   ├── Combined-P1-Dataset/         # Stage 1 Kaggle gem dataset (download separately)
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── splits/                      # Generated train/val/test CSVs (not in git)
└── results/                         # Training outputs and metrics
```

---

## Dataset Setup

All image datasets must be downloaded from Kaggle before training or evaluation.

### Stage 1 — Gem Species Dataset

Download the combined gemstone dataset from Kaggle and place it at:
```
data/Combined-P1-Dataset/
```
The folder should contain `train/`, `valid/`, and `test/` subdirectories with one subfolder per species (68 classes including Diamond).

### Stage 2 — Diamond Price Dataset

Download the JamesAllen and BrilliantEarth diamond datasets from the private Kaggle datasets (shared with team) and place images at:
```
ja_scraper/output/images/    # JamesAllen diamonds
be_scraper/output/images/    # BrilliantEarth diamonds
```
Images are organized as `images/{value_tier}/{diamond_id}.jpg`.

### Generate Splits

After placing the datasets, generate the train/val/test splits:
```bash
python scripts/build_combined_splits.py
python scripts/build_combined_dataset.py
```

This produces the CSV files in `data/splits/` that all training and evaluation scripts depend on.

---

## Installation

```bash
git clone https://github.com/JunyiiBlvd/481DLCGIPG.git
cd 481DLCGIPG

python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: .\venv\Scripts\activate   # Windows

pip install torch torchvision numpy pandas scikit-learn pillow
```

---

## Training

### Stage 1 — Gem Species Classifier

```bash
python src/train_stage1.py --arch efficientnetv2
python src/train_stage1.py --arch vit
```

Outputs to `results/training/stage1/{arch}/`.

### Stage 2 — Single-Site Classification

```bash
python src/train.py \
    --arch efficientnetv2 \
    --subset ja_natural \
    --data_dir ./data \
    --image_dir_ja ./ja_scraper/output/images \
    --image_dir_be ./be_scraper/output/images \
    --results_dir ./results \
    --epochs 30 \
    --batch_size 64
```

Subsets: `ja_natural`, `ja_lab`, `be_natural`, `be_lab`
Modes: add `--cross_domain` for cross-retailer evaluation

### Stage 2 — Single-Site Regression

```bash
python src/train_regression.py \
    --arch efficientnetv2 \
    --subset ja_natural \
    --data_dir ./data \
    --image_dir_ja ./ja_scraper/output/images \
    --image_dir_be ./be_scraper/output/images \
    --results_dir ./results
```

### Stage 2 — Combined Retailer (Classification)

```bash
python src/train_combined.py --arch efficientnetv2 --subset combined_all
```

Subsets: `combined_natural`, `combined_lab`, `combined_all`

### Stage 2 — Combined Retailer (Regression)

```bash
python src/train_regression_combined.py --arch efficientnetv2 --subset combined_all
```

### Random Forest Baseline

```bash
python src/rf_baseline.py
```

---

## Pipeline Evaluation

Runs the full two-stage pipeline on the held-out test set and reports:

- Stage 1 diamond recall (what percentage of diamonds are correctly identified)
- Stage 1 false positive rate (what percentage of non-diamonds are misidentified as Diamond)
- Stage 2 price tier accuracy and macro F1
- Stage 2 USD price mean absolute error and median absolute percentage error
- End-to-end accuracy per subset (ja_natural, ja_lab, be_natural, be_lab)

Requires:
- `results/training/stage1/efficientnetv2/best_model.pth`
- `results/training/regression/efficientnetv2/combined_all/best_model.pth`
- Both datasets in place and splits generated (see Dataset Setup)

```bash
python src/evaluate_pipeline.py
```

Results are saved to `results/pipeline_eval/pipeline_eval_results.json`.

---

## Model Architectures

| Architecture | Parameters | Stage 2 Head |
|---|---|---|
| ResNet-50 | ~25.6M | Linear(2048→512) → ReLU → Dropout → Linear(512→4) |
| EfficientNetV2-S | ~21.5M | Dropout → Linear(1280→4) |
| ViT-B/16 | ~86.6M | Dropout → Linear(768→4) |

All models use ImageNet-1K pretrained weights and differential learning rates (backbone: 0.1x head LR).

---

## Value Tier Classification

Diamonds are classified into 4 value tiers based on price distribution within each subset:

| Tier | Description |
|---|---|
| budget | Entry-level pricing |
| investment_grade | Top-tier pricing |
| mid_range | Mid-market pricing |
| premium | Above-average pricing |

Regression models predict continuous log-price (z-scored per subset) and use tier boundary thresholds computed from training data means.

---

## Dataset Statistics

| Source | Natural | Lab-grown | Total |
|---|---|---|---|
| JamesAllen.com | 107,687 | 121,519 | 229,206 |
| BrilliantEarth.com | 106,950 | 260,875 | 367,825 |
| Total | 214,637 | 382,394 | 597,031 |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | Cosine annealing (eta_min: 1e-6) |
| Batch size | 64 |
| Head learning rate | 3e-4 |
| Backbone learning rate | 3e-5 |
| Early stopping patience | 5 epochs (within-site), 10 epochs (combined) |
| Image size | 224x224 (ImageNet normalization) |

---

## Output Structure

```
results/training/{arch}__{subset}__{mode}/
├── best_model.pth            # Best checkpoint by val F1 (not in git)
├── train_log.json            # Per-epoch metrics
├── final_metrics.json        # Test evaluation results
└── classification_report.txt

results/training/regression/{arch}/{subset}/
├── best_model.pth            # Best checkpoint by val log-MAE (not in git)
├── train_log.json
└── final_metrics.json

results/pipeline_eval/
├── pipeline_eval_results.json  # Summary metrics
└── pipeline_eval_detail.csv    # Per-image predictions (not in git)
```

---

## References

1. He et al. (2016). Deep Residual Learning for Image Recognition
2. Tan & Le (2021). EfficientNetV2: Smaller Models and Faster Training
3. Dosovitskiy et al. (2021). An Image is Worth 16x16 Words
4. Chow & Reyes-Aldasoro (2023). Automatic Gemstone Classification Using Computer Vision

---

## Authors

| Name | Role | Email |
|---|---|---|
| Sebastian | Sections 3, 4 | scrimentis1@southernct.edu |
| Logan | Sections 5, 6 | caraballol2@southernct.edu |
| Shlok | Sections 1, 2 | gandhis2@southernct.edu |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

This project is for academic research purposes. Diamond images were collected from publicly accessible product pages and should not be used for commercial diamond grading without additional validation.
