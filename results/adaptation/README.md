# Domain Adaptation Results

Fine-tuning the EfficientNetV2 JA-natural cross-domain checkpoint on small BE samples
to bridge the JA→BE domain gap (baseline macro F1: 0.0708).

## Directory structure

```
adaptation/
  efficientnetv2__ja_natural__N{500,1000,2000}/   # original runs (max_epochs=10)
  efficientnetv2__ja_natural__N2000_ext/           # N=2000 extended to max_epochs=20
```

Each run directory contains:

| File | Description |
|---|---|
| `final_metrics.json` | Eval metrics on held-out remainder |
| `best_model.pth` | Adapted checkpoint (best FT-val F1) |
| `train_log.json` | Per-epoch train/val metrics |
| `classification_report.txt` | Per-class precision/recall/F1 |
| `sampled_ids.json` | The N images drawn from the pool (see format below) |
| `finetune_train.csv` | 80% of the N sample — used for gradient updates |
| `finetune_val.csv` | 20% of the N sample — used for early stopping |
| `eval.csv` | Held-out remainder of `be_natural_test.csv` — never seen during fine-tuning |

## sampled_ids.json format

```json
{
  "_format": "Keys are diamond_id (str), values are value_tier (str)",
  "75599509": "mid_range",
  "71046445": "mid_range",
  ...
}
```

`_format` is a documentation-only key (JSON has no comment syntax). Consumer scripts
should skip keys starting with `_`. value_tier is one of:
`budget` | `investment_grade` | `mid_range` | `premium`

## Pool

Source: `data/splits/be_natural_test.csv` (16,043 rows, site=brilliant_earth).
Sampling: stratified by `value_tier`, without replacement, `random_state=42`.
The three sets partition the pool exactly: `finetune_train ∪ finetune_val ∪ eval = pool`.

## Results summary

| Run | N fine-tune | Eval size | Macro F1 | Δ vs baseline |
|---|---:|---:|---:|---:|
| Baseline (no adaptation) | 0 | 16,043 | 0.0708 | — |
| N=500 | 400 train / 100 val | 15,543 | 0.2645 | +0.1937 |
| N=1000 | 800 / 200 | 15,043 | 0.2992 | +0.2284 |
| N=2000 | 1,600 / 400 | 14,043 | 0.3723 | +0.3015 |
| N=2000 ext (20 ep) | 1,600 / 400 | 14,043 | 0.4004 | +0.3296 |

## Stopping note

The N=2000 / 20-epoch run (F1=0.4004) had not converged at the epoch wall — FT-val F1
was still rising at epoch 19 and dropped only on epoch 20 due to patience expiry.
Based on the observed diminishing returns across epochs 15–20, the estimated asymptote
is approximately 0.42–0.43. No further epoch extension was performed; the marginal gain
does not justify additional compute at this stage of the project.

**Paper disclosure (known limitation):** The reported F1=0.4004 for the N=2000 adapted
model is not a converged result. The true performance ceiling with this sample size and
architecture is likely ~0.42–0.43. This should be stated explicitly when reporting
adaptation results to avoid overstating the difficulty of bridging the domain gap.
