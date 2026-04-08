# DLCGIPG — Deep Learning Classifiers for Gemstone Identification and Price Grading

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A two-stage deep learning pipeline for gemstone analysis: **Stage 1** classifies gemstone types, and **Stage 2** predicts diamond value tiers using visual features and structured 4C attributes.

> **CSC-481 Project** — Southern Connecticut State University

> **------------------WORK IN PROGRESS-----------------------**


## 📋 Overview

This project systematically compares three deep learning architectures:
- **ResNet-50** (He et al., 2016)
- **EfficientNetV2-S** (Tan & Le, 2021)
- **Vision Transformer (ViT-B/16)** (Dosovitskiy et al., 2021)

### Key Contributions
1. **Large-scale self-collected dataset**: 597,029 labeled diamond images from JamesAllen.com and BrilliantEarth.com
2. **Cross-domain generalization experiment**: Models trained on one retailer's data evaluated on the other's
3. **Visual vs. tabular comparison**: Deep learning classifiers benchmarked against Random Forest on 4C attributes

## 📁 Project Structure

```
481DLCGIPG/
├── src/                      # Core ML code
│   ├── models.py             # Model architectures (ResNet50, EfficientNetV2, ViT)
│   ├── train.py              # Classification training loop
│   ├── train_regression.py   # Price regression variant
│   ├── diamond_dataset.py    # PyTorch Dataset for diamond images
│   ├── rf_baseline.py        # Random Forest tabular baseline
│   ├── evaluate.py           # Model evaluation utilities
│   └── aggregate_results.py  # Results aggregation
├── ja_scraper/               # JamesAllen.com scraper
├── be_scraper/               # BrilliantEarth.com scraper
├── data/                     # Dataset splits and class weights
│   └── splits/               # Train/val/test CSV files
├── scripts/                  # Training and evaluation scripts
│   ├── run_within_site.sh    # Within-site training
│   ├── run_cross_domain.sh   # Cross-domain experiments
│   └── run_regression.sh     # Regression training
├── results/                  # Training outputs and metrics
├── docs/                     # Project documentation
└── Phase1-Combined-Dataset/  # Stage 1 gemstone datasets
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/JunyiiBlvd/481DLCGIPG.git
cd 481DLCGIPG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn pillow
```

### Training

**Within-site classification:**
```bash
python src/train.py \
    --arch resnet50 \
    --subset ja_natural \
    --data_dir ./data \
    --image_dir_ja ./ja_scraper/output/images \
    --image_dir_be ./be_scraper/output/images \
    --results_dir ./results \
    --epochs 30 \
    --batch_size 64
```

**Cross-domain experiment:**
```bash
python src/train.py \
    --arch efficientnetv2 \
    --subset ja_natural \
    --cross_domain \
    --data_dir ./data \
    --image_dir_ja ./ja_scraper/output/images \
    --image_dir_be ./be_scraper/output/images \
    --results_dir ./results
```

**Random Forest baseline:**
```bash
python src/rf_baseline.py
```

## 🏗️ Model Architectures

| Architecture | Parameters | Head Configuration |
|--------------|------------|-------------------|
| ResNet-50 | ~25.6M | Linear(2048→512) → ReLU → Dropout → Linear(512→4) |
| EfficientNetV2-S | ~21.5M | Dropout → Linear(1280→4) |
| ViT-B/16 | ~86.6M | Dropout → Linear(768→4) |

All models use:
- ImageNet-1K pretrained weights
- Differential learning rates (backbone: 0.1× head LR)
- Label smoothing (0.1)
- Weighted cross-entropy loss for class imbalance

## 📊 Value Tier Classification

Diamonds are classified into 4 value tiers based on price percentiles:

| Tier | Price Percentile | Description |
|------|-----------------|-------------|
| Budget | 0-25% | Entry-level diamonds |
| Mid-range | 25-50% | Average market price |
| Premium | 50-75% | Above-average quality |
| Investment Grade | 75-100% | Top-tier diamonds |

## 📈 Dataset Statistics

| Source | Natural | Lab-grown | Total |
|--------|---------|-----------|-------|
| JamesAllen.com | 107,687 | 121,519 | 229,206 |
| BrilliantEarth.com | 106,950 | 260,875 | 367,825 |
| **Total** | **214,637** | **382,394** | **597,031** |

## 🔧 Training Configuration

Default hyperparameters:
- **Optimizer**: AdamW (weight decay: 1e-4)
- **Scheduler**: Cosine annealing (η_min: 1e-6)
- **Batch size**: 64
- **Learning rate**: 3e-4 (head), 3e-5 (backbone)
- **Early stopping**: 5 epochs patience on validation F1
- **Image size**: 224×224 (ImageNet normalization)

## 📂 Output Structure

Training produces:
```
results/training/{arch}__{subset}__{mode}/
├── best_model.pth           # Best checkpoint (by val F1)
├── train_log.json           # Per-epoch metrics
├── final_metrics.json       # Test evaluation results
└── classification_report.txt # Detailed classification report
```

## 🧪 Experiments

### Within-Site Evaluation
Models trained and evaluated on the same retailer's data:
- `ja_natural`, `ja_lab` (JamesAllen)
- `be_natural`, `be_lab` (BrilliantEarth)

### Cross-Domain Evaluation
Models trained on one retailer, evaluated on the other:
- Train: JA → Test: BE
- Train: BE → Test: JA

This tests whether models learn transferable diamond quality features or site-specific imaging artifacts.

## 📚 References

1. He et al. (2016). "Deep Residual Learning for Image Recognition"
2. Tan & Le (2021). "EfficientNetV2: Smaller Models and Faster Training"
3. Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words"
4. Chow & Reyes-Aldasoro (2023). "Automatic Gemstone Classification Using Computer Vision"

## 👥 Authors

| Name | Role | Email |
|------|------|-------|
| Sebastian | Sections 3, 4 | scrimentis1@southernct.edu |
| Logan | Sections 5, 6 | caraballol2@southernct.edu |
| Shlok | Sections 1, 2 | gandhis2@southernct.edu |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for academic research purposes. Diamond images were collected from publicly accessible product pages. The models are trained for research evaluation and should not be used for commercial diamond grading without additional validation.
