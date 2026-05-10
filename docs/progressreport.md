CSC-481
DLCGIPG
Deep Learning Classifiers for Gemstone Identification and Price Grading



Project Progress Report

April 3, 2026

Logan Caraballo
33% Contribution
Stage 2 training · GPU
Sebastian Scrimenti
33% Contribution
Stage 1 training · GPU
Shlok Gandhi
33% Contribution
Evaluation · Report infra



Southern Connecticut State University
caraballol2@southernct.edu  ·  scrimentis1@southernct.edu  ·  gandhis2@southernct.edu


























1. Executive Summary
This report documents the current state of DLCGIPG (Deep Learning Classifiers for Gemstone Identification and Price Grading), a two-stage machine learning pipeline designed to evaluate whether visual models can infer gemstone identity and diamond value from retail photography.
As of April 7, 2026, Stage 2 experimentation is fully complete, including:
12 within-site models (3 architectures and 4 subsets)
12 cross-domain models (3 architectures x 2 directions x 2 subsets per direction)
Domain adaptation experiments (EfficientNetv2-S, N=500/1000/2000 / 2000-ext)
A regression-based reformulation of the task
Across all experiments, EfficientNetV2-S is the strongest architecture, outperforming ResNet50 and ViT-B/16 consistently across datasets. Within-site performance ranges from 0.53 - 0.67 macro F1, while the Random Forest tabular baseline reaches 0.82 - 0.92, quantifying the gap between visual and structured data.
The primary finding is that cross-domain transfer fails. Models trained on JamesAllen (JA) collapse when evaluated on BrilliantEarth (BE) (mean F1 = approx. 0.12), while BE to JA transfer performs moderately better (approx. 0.24), demonstrating strong domain asymmetry.
A domain adaptation experiment shows this gap is partially recoverable: fine-tuning with 2000 target-domain samples improves F1 from 0.07 to 0.40.
A regression experiment predicting log(price) instead of discrete tiers confirms that images contain meaningful continuous value signals, achieving ~25% median absolute percentage error across domains.
The results establish three core conclusions:
CNN architectures outperform transformers at this dataset scale
Domain shift is the dominant limitation in real-world deployment
Visual features encode value signal, but not at tabular precision

Component
Status
Notes
Stage 2 Data collection & cleaning
Complete
492,643 labeled records, 492,257 images
Kaggle upload
Complete
ja-diamond-images-4c v2 · be-diamond-images-4c v1
Train/val/test splits
Complete
Locked seed 42, stratified, 20 split CSVs
RF baseline (classification)
Complete
0.82 - 0.92 F1 performance ceiling established
Training infrastructure
Complete
models.py, train.py, evaluate.py, launchers
Stage 2 within-site (12/12)
Complete
All 3 architectures x 4 subsets
Stage 2 cross-domain (12/12)
Complete
All 3 architectures x 4 direction/subset combos
Domain adaptation
Complete
EfficientNetV2, N=500/1000/2000/2000-ext
Regression experiment
Complete
ja_natural & be_natural, log(price) target
RF regression baseline
Complete
Ceiling for regression comparison
Seed Validation
Pending
Results pending
Cross-domain regression
Pending
Results pending
Stage 1 dataset label normalization
Complete


Stage 1 training
Pending
Results pending
End-to-end pipeline eval
Pending
Dataset construction approach TBD with professor

2. Project Overview
2.1 Research Questions
The project addresses five formal research questions:
Can ResNet50, EfficientNetV2, or ViT-B/16 classify gemstone types accurately on the MDPI Minerals benchmark, and does a larger training set reverse the finding that ResNet-50 underperforms Random Forest on ~2,000 images?
Can the same architectures classify diamond value tiers from retail photography at accuracy levels that exceed a random baseline?
Does any visual architecture match or exceed the tabular 4C Random Forest baseline, or does the information gap between image and structured attributes prevent visual classifiers from reaching tabular performance levels?
Do models trained on JamesAllen.com images generalize to BrilliantEarth.com images, and vice versa? What cross-domain accuracy delta is observed?
Does model performance differ between natural and lab-grown diamond subsets?


2.2 Pipeline Architecture
A single retail diamond photograph enters the pipeline. Stage 1 classifies the image by gemstone type. If the image is classified as a diamond it is passed to Stage 2 which predicts one of four value tiers. Non-diamond images are rejected at Stage 1. The diagram below illustrates this flow.


















Diagram 1 — Two-stage pipeline flow


3. Datasets
3.1 Stage 1 Datasets
Both Stage 1 datasets were sourced from Kaggle and were publicly available for research use. Data sets were trimmed according to "Automatic Gemstone Classification Using Computer Vision" by Bona Hiu Yan Chow and Constantino Carlos Reyes-Aldasoro which featured classifications based on a narrowed subset of Daria Chemkaeva’s Gemstones Images dataset. Stage 1 Images were relabelled and unified to fit a singular naming convention.

Dataset
Author
Source
Images
Classes
Precious Gemstone Identification
GauravKamath02
Kaggle
49,273
87 (Trimmed 68)
Gemstones Images
Daria Chemkaeva
Kaggle
2.326
87 (Trimmed 68)
TOTAL




51,599









3.2 Stage 2 Self-Collected Datasets
Both Stage 2 datasets were collected via custom Python scrapers targeting publicly accessible product pages. No authentication was required. Scrapers identified themselves as academic research tools.
Dataset
Source
Rows
Images
Shapes
Kaggle
JA Natural
JamesAllen.com
107,687
107,687
10
ja-diamond-images-4c v2
JA Lab-Grown
JamesAllen.com
121,517
121,517
10
ja-diamond-images-4c v2
BE Natural
BrilliantEarth.com
106,950
106,950
10
be-diamond-images-4c v1
BE Lab-Grown
BrilliantEarth.com
156,487
156,103
10
be-diamond-images-4c v1
TOTAL
-
492,641
492,257
-
-


3.3 Value Tier Boundaries (JA Natural)
Tier
Boundary
Count
% of Total
Budget
≤ $840
59,427
25.9%
Mid-Range
$840 - $4,440
110,868
48.3%
Premium
$4,440 - $11,880
34,057
14.8%
Investment-Grade
> $11,880
24,854
10.8%

Class imbalance ratio ~4.5x. Addressed via WeightedRandomSampler and weighted cross-entropy loss. Macro F1 reported alongside accuracy.


3.4 Key Dataset Characteristics
JA photographs every diamond at 40x magnification under standardized lighting most controlled consumer-accessible diamond photography available
BE sources diamonds from multiple suppliers with varied imaging setups intentionally less homogeneous, basis of cross-domain experiment
BE round diamond gap remediated: 104,495 round images added after merge bug discovered in original scrape
Cut vocabulary mismatch resolved: GIA 'Excellent' (JA) and AGS 'Super Ideal' (BE) remapped to unified 'Ideal+' label
BE tier boundaries for round diamonds computed from non-round inventory (9 shapes) disclosed limitation
Lab-grown and natural diamonds treated as separate subsets throughout different price spaces, results reported independently




3.5 Dataset Lineage
Diagram 2 - Dataset lineage


4. Random Forest Baseline
4.1 Purpose and Design
The Random Forest baseline serves as the performance ceiling for Stage 2 image models. Unlike the deep learning models which receive only a photograph, the RF receives the four structured 4C attributes directly: carat weight, cut grade, color grade, and clarity grade. These are the exact attributes professional gemologists use to price diamonds. The RF therefore has near-perfect information relative to the tier labels, which were derived from price using these same attributes.

The baseline answers two questions: (1) are the tier labels coherent and learnable, and (2) what is the upper bound for attribute-driven prediction? Image models cannot access carat weight from a photograph; they must infer it visually. The gap between RF performance and image model performance quantifies this information loss.


4.2 Results
Subset
Test Accuracy
Macro F1
Carat Importance
ja_natural
0.8658
0.8567
0.853
ja_lab
0.9411
0.9217
0.793
be_natural
0.8371
0.8226
0.805
be_lab
0.8961
0.8742
0.784


4.3 Interpretation
Carat dominance (0.79 - 0.85 feature importance) is expected diamond price scales exponentially with carat weight. This is not a labeling artifact.
The remaining 15 - 21% signal from color, clarity, and cut confirms the labels reflect real pricing structure, not pure carat bucketing.
JA lab (F1 = 0.922) is the easiest subset, lab diamonds have tighter grade distributions and more predictable pricing by carat.
These numbers are the ceiling. The research question is how close image models get, not whether they exceed it.


5. Model Architectures
All three architectures were pretrained on ImageNet-1K. The final classification head was replaced with a 4-class output layer. All parameters are fine-tuned, not frozen feature extraction.

Architecture
Parameters
Backbone
Head Replacement
ResNet50
24.6M
ImageNet1K_V2
Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→4)
EfficientNetV2-S
20.2M
ImageNet1K_V1
Dropout(0.3) → Linear(1280→4)
ViT-B/16
85.8M
ImageNet1K_V1
Dropout(0.3) → Linear(768→4)
















Diagram 3 — Model architecture comparison

5.1 Training Hyperparameters
Parameter
Value
Rationale
Optimizer
AdamW
Adaptive LR with decoupled weight decay
Base LR (head)
3e-4
Head initialized randomly - needs larger steps
Backbone LR
3e-5 (base x 0.1)
Pretrained weights - small updates only
Scheduler
CosineAnnealingLR
Smooth decay from 3e-4 to 1e-6 over 30 epochs
Batch size
64
Conservative for 16GB VRAM - consistent across all runs
Epochs (max)
30
With early stopping (patience = 5) on val macro F1
Label smoothing
0.1
Prevents overconfidence, improves generalization
Gradient clipping
max_norm = 1.0
Prevents exploding gradients
Dropout (head)
0.3
Regularization in classification head
AMP
float16 + GradScaler
~2x throughput on RTX 5070 Ti CUDA cores
Class weights
Per-split weighted CE loss
Corrects ~4.5x imbalance between mid_range and investment_grade


Cross-domain runs only: AMP was disabled for all 12 cross-domain runs. Float16 caused NaN loss at multiple epochs across multiple attempts due to gradient spikes on the cross-domain data distribution. All cross-domain runs used float32. Max epochs increased to 50, patience increased to 10 for cross-domain runs.

6. Training Loop
Each epoch alternates between a training phase and a validation phase. Test evaluation occurs exactly once per run after training completes, using the best checkpoint selected by validation macro F1.


7. Stage 2 Within - Site Results
All 12 within site runs are complete

7.1 Results Table
Architecture
Subset
Test Accuracy
Macro F1
Best Val F1
Epochs Run
ResNet50
ja_natural
0.6652
0.6590
0.6620
25 (early stop)
ResNet50
ja_lab
0.6674
0.6386
0.6410
30 (max)
ResNet50
be_natural
0.6289
0.6075
0.6042
30 (max)
ResNet50
be_lab
0.5875
0.5557
0.5556
30 (max)
EfficientNetV2
ja_natural
0.6798
0.6724
0.6845
28 (early stop)
EfficientNetV2
ja_lab
0.6904
0.6589
0.6665
30 (max)
EfficientNetV2
be_natural
0.6292
0.6093
0.6117
30 (max)
EfficientNetV2
be_lab
0.5822
0.5554
0.5574
30 (max)
ViT-B/16
ja_natural
0.5411
0.5595
0.5648
12 (early stop)
ViT-B/16
ja_lab
0.617
0.6181
0.6239
30(max)
ViT-B/16
be_natural
0.6326
0.5885
0.5917
30(max)
ViT-B/16
be_lab
0.5786
0.5295
0.5312
30 (max)


7.2 Per-Class F1
Architecture
Subset
Budget F1
Mid-Range F1
Premium F1
Inv-Grade F1
ResNet50
ja_natural
0.787
0.495
0.706
0.648
ResNet50
ja_lab
0.807
0.486
0.590
0.671
ResNet50
be_natural
0.743
0.446
0.610
0.632
ResNet50
be_lab
0.718
0.393
0.494
0.617
EfficientNetV2
ja_natural
0.783
0.528
0.714
0.665
EfficientNetV2
ja_lab
0.814
0.505
0.618
0.699
EfficientNetV2
be_natural
0.752
0.451
0.609
0.624
EfficientNetV2
be_lab
0.722
0.400
0.505
0.595
ViT-B/16
ja_natural
0.739
0.418
0.664
0.418
ViT-B/16
ja_lab
0.782
0.436
0.566
0.689
ViT-B/16
be_natural
0.729
0.398
0.556
0.671
ViT-B/16
be_lab
0.692
0.342
0.455
0.629

7.3 Results Hierarchy
Diagram 5 - Results performance hierarchy

8. Key Findings
8.1 EfficientNetV2-S Leads Both CNNs
EfficientNetV2-S outperforms ResNet50 on every completed subset by 0.1 to 1.3 F1 points with fewer parameters (20.2M vs 24.6M). Despite being the lightest architecture, compound scaling delivers better accuracy per parameter at this dataset scale. For the paper this is a meaningful finding the smallest model wins.
8.2 ViT-B/16 Significantly Underperforms
ViT-B/16 early stopped at epoch 12 on ja_natural (F1 = 0.560) while ResNet50 and EfficientNetV2 ran to 25 - 30 epochs and reached 0.659 - 0.672. ViT requires substantially more data than CNNs to generalize, as even at 75,000 training images it is data-starved relative to what ViT-B/16 needs to beat CNNs. This extends the Chow et al. (2022) finding: scale alone does not reverse the CNN vs transformer ordering when the dataset is not sufficiently large. ViT epoch times also increased during training (184s down to 231s), indicating CPU thermal throttling on DataLoader workers during the long run.


8.3 mid_range is the Hardest Class - Pre-Registered Prediction Confirmed
mid_range produced the lowest per-class F1 in every single completed run across both architectures. This was predicted before training began from first principles: mid_range is the largest class (48% of samples), sits between two adjacent classes with the least price separation from budget and premium, and has the highest within-class carat variance. Confirmation from the actual results before cross-domain runs strengthens this as a paper finding.
8.4 BE is Consistently Harder Than JA
BE subsets show 5 - 8 F1 points lower performance than corresponding JA subsets across both architectures. JA's standardized 40x superzoom regime produces visually homogeneous images. BE's multi-supplier imaging produces more varied backgrounds, resolutions (300x300 to 460x460), and lighting conditions. A model trained on BE must generalize across a wider visual distribution. This imaging difference is the plausible explanation and will be tested directly by the cross-domain experiment.
8.5 Image-to-Tabular Gap Quantified
The gap between image model performance and the RF tabular ceiling is 0.18 - 0.27 F1 points across all completed subsets. This quantifies the cost of working from retail photographs instead of structured 4C attributes. The gap exists because a JPEG cannot communicate carat weight (0.79 - 0.85 RF feature importance) with the precision of a scale measurement. This is the paper's answer to Research Question 3: no visual architecture approaches the tabular ceiling, and the gap is explained by the fundamental information difference between the two modalities.
8.6 investment_grade: High Precision, Low Recall
Across all completed runs, investment_grade shows high precision (0.77 - 0.87) but low recall (0.52 - 0.55). The model is conservative: when it predicts investment_grade it is usually correct, but it misclassifies more than half of actual investment_grade diamonds as mid_range or premium. Investment_grade diamonds represent only 10.8% of the dataset. Even with class weighting, the model hedges toward larger classes when uncertain.

9. Cross-Domain Experiment Design
All 12 cross-domain runs are complete. Models were trained on one retailer’s full dataset and evaluated on the other retailer’s held test set.
Diagram 6 - Cross-domain experiment design

9.1 Cross-Domain Results
Architecture
Train
Test
Direction
Test Acc
Macro F1
ResNet50
ja_natural
BE
JA to BE
0.1719
0.1538
ResNet50
ja_lab
BE
JA to BE
0.1475
0.1284
ResNet50
be_natural
JA
BE to JA
0.3914
0.3097
ResNet50
be_lab
JA
BE to JA
0.2709
0.2048
EfficientNetV2
ja_natural
BE
JA to BE
0.1089
0.0708
EfficientNetV2
ja_lab
BE
JA to BE
0.1550
0.1371
EfficientNetV2
be_natural
JA
BE to JA
0.4216
0.3054
EfficientNetV2
be_lab
JA
BE to JA
0.1684
0.1411
ViT-B/16
ja_natural
BE
JA to BE
0.1145
0.0872
ViT-B/16
ja_lab
BE
JA to BE
0.1467
0.1286
ViT-B/16
be_natural
JA
BE to JA
0.2604
0.2162
ViT-B/16
be_lab
JA
BE to JA
0.3576
0.2424


9.2 Direction Summary
Direction
Mean Macro F1
Min
Max
JA to BE
0.1177
0.0708
0.1538
BE to JA
0.2366
0.1411
0.3097

9.3 Key Findings
JA to BE failure is universal across architectures.
All six JA to BE runs failed across all three architectures (mean F1=0.12). Three different model families, two subsets each and consistent collapse. Budget class almost vanishes (F1=0.0005–0.033). Failure is distributed across all classes rather than collapsing to a single prediction mode in the final corrected runs.
BE to JA shows partial transfer.
Mean BE to JA F1=0.24, approximately 2x JA to BE. BE's multi-supplier imaging produces more generalizable visual features than JA's controlled 40x regime. Natural subsets transfer better than lab subsets: BE natural to JA reaches F1=0.31 (ResNet50, EfficientNetV2) while BE lab to JA drops to 0.14–0.24.
investment_grade transfers most reliably 
Across BE to JA runs (F1=0.37–0.58). We see large, visually distinctive stones image consistently regardless of source. Budget fails most severely in JA to BE (F1 near zero).
9.4 Domain Shift Quantification
Metric
JA natural
BE natural
Cohen's d
R channel mean
182
174
0.77 (large)
G channel mean
-
-
0.61 (medium)
B channel mean
-
-
0.28 (small)
Channel std (contrast)
~43
~36
-
Resolution
757×600 (99.7% fixed)
~300×300 (variable 215–460px)
-
Aspect ratio
1.262 ± 0.000
1.028 ± 0.082
-

Cohen’s d = difference in means / pooled standard deviation
Cross-domain failure is categorical, not continuous. The EfficientNetV2 JA natural cross-domain model predicts "premium" for 93.7% of BE inputs (0 budget, 19 investment_grade, 108 mid_range, 1873 premium). The brightness shift (Cohen's d=0.77 on R channel) and 2.5x resolution difference displace BE inputs outside the model's learned decision space entirely.
10. Domain Adaption Experiment
10.1 Design
The best within-site architecture (EfficientNetV2-S) and worst JA→BE result (F1=0.0708, JA natural) were selected for maximum contrast. Fine-tuning pool: be_natural_test.csv (16,043 images — never seen during original training). Configuration: LR=1e-5, frozen backbone except last two blocks + classifier head, float32, patience=3.
Leakage verified: ft_train ∩ ft_val = 0, ft_train ∩ eval = 0, ft_val ∩ eval = 0. ft_train + ft_val + eval = 16,043 = pool total for all N. Exact partition confirmed.
10.2 Results
Run
Fine-tune N
Epochs
Eval Macro F1
vs Baseline
Multiplier
Baseline (zero-shot)
0
-
0.0708
-
1.0x
N=500
400 train / 100 val
10
~0.261
+0.190
3.7x
N=1000
800 / 200
10
~0.320
+0.249
4.5x
N=2000
1600 / 400
10
0.3723
+0.302
5.3x
N=2000 ext
1600 / 400
20
0.4004
+0.330
5.7x

Gains are monotonic with N and epochs. Budget and investment_grade recover fastest. N=2000 ext was not converged at epoch 20 (best epoch 19); estimated asymptote ~0.42–0.43. The domain gap is bridgeable; we now see that this is a calibration problem, not a representational one.
11. Regression Experiment an Alternative Formulation
11.1 Design
EfficientNetV2-S head replaced: Linear(1280→4) → Linear(1280→1). Loss: HuberLoss(delta=0.5). Target: log(price_usd). Subsets: ja_natural and be_natural within-site only.
11.2 Training Outcomes
Run
Best Val log-MAE
Best Epoch
Epochs Run
Converged
ja_natural
0.3300
14
19
Yes
be_natural
400 train / 100 val
28
30
No


11.3 Vision vs RF Regression Baseline
Metric
RF (JA)
Vision (JA)
RF (BE)
Vision (BE)
log-MAE
0.153
0.332
0.138
0.347
USD-MAE
$976
$2,184
$665
$1,721
Median APE %
11.5%
26.0%
10.6%
25.5%
R² (log)
-
0.880
-
0.783
Spearman ρ
-
0.931
-
0.885



11.4 Regression-to-Tier Bridge
Subset
Regression Tier F1
Direct Classification F1
Delta
ja_natural
0.7366
0.6724
+0.064
be_natural
0.6906
0.6093
+0.081

Regression-derived classification outperforms direct classification on both within-site subsets tested, suggesting it may be a more effective formulation. The continuous target preserves ordinal structure that 4-class cross-entropy loses near tier boundaries. Cross-domain regression results pending overnight runs.
12. Team Responsibilities
12.1 Logan Caraballo - Stage 2 Training
Scraped JA dataset (229,204 images) via GraphQL API micro-band strategy
Contributed to BE dataset scraping
Built full preprocessing pipeline: cleaned CSVs, splits, class weights
Ran RF baseline establishing performance ceiling
Built all training infrastructure: models.py, train.py, evaluate.py, launchers
Running all 12 within-site training runs on protouno GPU
Will run 12 cross-domain runs after within-site completes
Responsible for aggregate_results.py final summary table generation
12.2 Sebastian Crimentis - Stage 1 Training
Stage 1 gemstone classifier using Sindhu + Kamath + Purswani datasets on Colab
Label vocabulary normalization across Sindhu and Kamath before any training code is written
Implement RF baseline replicating Chow et al. (RGB histogram + LBP) for direct literature comparison
Train ResNet50, EfficientNetV2, ViT-B/16 on merged Stage 1 training pool
Evaluate all three architectures against MDPI Minerals benchmark (held out - never trained on)
Existing train.py/models.py on GitHub is reusable. Only the Dataset class needs replacing
12.3 Shlok Gandhi - Evaluation and Report
Stage 1 evaluation scripts: confusion matrix, per-class F1, architecture comparison table 
Adapting evaluate.py from GitHub for gemstone classes (swap 4 diamond tier labels for gemstone list)
Proposal CSC481_Proposal_v4.md needs immediate updates: corrected dataset numbers, BE tier boundary disclosure, Team Responsibilities section
Report template and section infrastructure 
Final paper tables: Stage 1 architecture vs Chow baseline, Stage 2 within-site, cross-domain delta
13. Next Steps
13.1 Immediate (Logan)
Collect overnight seed validation results and cross-domain regression results
13.2 Immediate (Sebastian)
Look into image segmentation and masking to improve data quality
13.3 Immediate (Shlok)
Update CSC481_Proposal_v4.md
Begin report structure -  section headers, methodology template, figure placeholders
13.4 Upcoming - End-to-End Evaluation
The end-to-end pipeline evaluation dataset does not yet exist. Options discussed:
Construct from existing data: combine MDPI non-diamond images with a sample from JA/BE test splits -  provides both real gating decisions and measurable Stage 2 accuracy
Theoretical calculation: Stage 1 diamond recall x Stage 2 accuracy - valid but weaker than empirical measurement
Raise with professor before building - minor scope addition but significantly strengthens pipeline credibility claim


DLCGIPG  ·  CSC-481 Southern Connecticut State University

14. References
[1] K. He, X. Zhang, S. Ren, and J. Sun,
“Deep Residual Learning for Image Recognition,”
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
https://arxiv.org/abs/1512.03385
[2] M. Tan and Q. V. Le,
“EfficientNetV2: Smaller Models and Faster Training,”
International Conference on Machine Learning (ICML), 2021.
https://arxiv.org/abs/2104.00298
[3] A. Dosovitskiy et al.,
“An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,”
International Conference on Learning Representations (ICLR), 2021.
https://arxiv.org/abs/2010.11929
[4] L. Sindhu, “Gemstones Images Dataset,” Kaggle. 
Available: 
https://www.kaggle.com/datasets/lsind18/gemstones-images
[5] G. Kamath, “Precious Gemstone Identification Dataset,” Kaggle. 
Available: https://www.kaggle.com/datasets/gauravkamath02/precious-gemstone-identification
[6] L. Caraballo, S. Crimentis, S. Gandhi,
“BE Diamond Images — 4C Value Tiers (263K),” Kaggel.
Available:
https://www.kaggle.com/datasets/junyiiblvc/be-diamond-images-4c
[7] L. Caraballo, S. Crimentis, S. Gandhi, 
“JA Diamond Images — 4C Value Tiers (229K),” Kaggle.
Available:
https://www.kaggle.com/datasets/junyiiblvc/ja-diamond-images-4c
[8] S. Bansal, “Diamonds Dataset,” Kaggle. 
Available:
https://www.kaggle.com/datasets/shivam2503/diamonds  
[9] A. Purswani, "Diamond Images Dataset," Kaggle. Retrieved March 2025. 
Available: 
https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset 
[10] H. Lakhani, "Natural Diamonds Prices + Images," Kaggle. Retrieved March 2025. Available: 
https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images 
[11] Mo Zhou, “Enhancing Diamond Price Prediction through Machine Learning and Deep Learning: A Comparative Analysis of AGS and GIA Grading Systems”, 2025

