# Cross-Primitive Consistency in Jev

Official code for **[Cross-Primitive Consistency in Jev](https://arxiv.org/abs/2610.38623)**.

[![arXiv](https://img.shields.io/badge/arXiv-2610.38623-b31b1b.svg)](https://arxiv.org/abs/2610.38623)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E.svg)](https://huggingface.co/datasets/67L11/PrimaCon)
[![Calibrators](https://img.shields.io/badge/Calibrators-Hugging%20Face-FFD21E.svg)](https://huggingface.co/67L11/CPRC)

## Overview

Does a fast decision model give compatible answers when the same state is expressed through different typed queries? **PrimaCons** evaluates complement, label hierarchy, abstention alignment, and label-description consistency. **CPRC**, the Cross-Primitive Residual Calibrator, adds a learned logit correction to a frozen JEV backbone.

---

## Installation

### Step 1: Create an environment

```bash
conda create -n primaconsis python=3.10 -y
conda activate primaconsis
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 1. Prepare Data and Models

### Step 1: Download the benchmark

- URL: [67L11/PrimaCons](https://huggingface.co/datasets/67L11/PrimaCons)
- Destination: `data/primacon/`

```bash
hf download 67L11/PrimaCons --repo-type dataset --local-dir data/primacons
```

Each record contains a state, labels and source metadata.

```text
data/primacon/
├── axis1/{calibration,test}.jsonl
├── axis2/{calibration,test}.jsonl
├── axis3/{calibration,test}.jsonl
└── axis4/{calibration,test}.jsonl
```

| Axis | Perturbation | Source | Calibration | Test |
|---|---|---|---:|---:|
| I | Positive/negative Noul and binary Choice | HaluEval dialogue | 6,000 | 6,000 |
| II | Coarse/medium/fine Choice and same-level Noul | PKU-SafeRLHF | 6,000 | 6,000 |
| III | Answer selection versus answerability | AbstentionBench | 5,898 | 5,837 |
| IV | Raw/described labels and canonical Noul | AG News | 6,000 | 6,000 |


### Step 2: Download calibrators

- URL: [67L11/CPRC](https://huggingface.co/67L11/CPRC)
- Destination: `models/calibrators/`

```bash
hf download 67L11/CPRC --local-dir models/calibrators
```

Each `2b/`, `4b/`, and `9b/` folder has `calibrator.pt`, `pca.pkl`, and `config.json`. Always pair the checkpoint with its own PCA.

### Step 3: Download a JEV backbone

```bash
# Default: 4B
hf download alibiserikbay/JevK5 --local-dir models/JevK5
# Optional alternatives: update config.yaml too
hf download alibiserikbay/JevK5-2B --local-dir models/JevK5-2B
hf download alibiserikbay/JevK5-9B --local-dir models/JevK5-9B
```

Use the matching experiment revision, not an arbitrarily updated upstream `main`. Calibrator weights do not include the backbone.

### Step 4: Configure paths

```yaml
model_size: 4b
feature_recipe: 4b
paths:
  dataset: data/primacons
  backbone: models/JevK5
  calibrator: models/calibrators/4b
  output: outputs/4b
  trained: outputs/4b/trained
training:
  source_axes: [D1, D2, D3]
  pca_dim: 64
  alpha: 1.4
  gamma: -1.0
  max_epochs: 10000
  patience: 5
```

Paths resolve relative to `config.yaml`. For 2B/9B change the size, feature recipe and model/output paths together. The recipe preserves checkpoint-specific historical feature details.

---

## 2. Real-Time Calibrated Inference

Load once and pass any task as Choice or Noul. JEV selects options or returns truth probabilities.

```python
from primaconsis import config
from online import CalibratedJev

model = CalibratedJev(config("config.yaml"))
result = model.decide("Question: What is the capital of France?", {
    "type": "choice", "instructions": "Select the correct answer.",
    "criteria": {"Paris": "", "Berlin": "", "Rome": ""},
})
print(result["choice"], result["probabilities"])
result = model.decide("Paris is the capital of France.", {
    "type": "noul", "instructions": "Is the statement correct?",
})
print(result["noul"])
```

Each call uses one backbone forward and immediately calibrates using its hidden state.


## 3. Run and Evaluate the Entire Benchmark

Run on the GPU after downloading data and models. The runner visits all four axes.

```bash
python run_benchmark.py --config config.yaml --split test --output test_predictions.jsonl
python evaluate_benchmark.py --input test_predictions.jsonl --output test_metrics.json

# Calibration diagnostics (Tables 1–4)
python run_benchmark.py --config config.yaml --split calibration --output calibration_predictions.jsonl
python evaluate_benchmark.py --input calibration_predictions.jsonl --output calibration_metrics.json
```

The runner saves both native JEV and CPRC predictions.

`evaluate_benchmark.py` contains all four axis metric definitions, and saves separate `raw` and `calibrated` results:

| Axis | Paper metrics |
|---|---|
| I | Three mean absolute probability gaps; positive Noul, reversed Noul and Choice accuracy |
| II | Coarse/medium/fine mean TV; Choice accuracy at each level |
| III | Mean abstention gap, hard decision conflict, accuracy gap and both accuracies |
| IV | Three mean TV distances; raw/described Choice accuracy |


---

## 4. Train on Source Calibration Only

### Step 1: Select source axes

Set `training.source_axes`, e.g. `[D1, D2, D3]` to hold out D4. Only source calibration records enter PCA, standardization, optimization and early stopping. There is no target-calibration injection or alternative training-mode switch.

### Step 2: Train

```bash
export TRC_TRAIN_DEVICE=cuda  # or cpu
export TRC_THREADS=4
python primaconsis.py train --config config.yaml
```

The residual objective combines Gold-weighted CE and metric-aligned consistency. Defaults: alpha 1.4, gamma -1, delta_max 1, Adam lr 0.002, weight decay 1e-4. Source families are hash-split 60/20/20 into train/select/audit. PCA fits train only; select controls patience-5 early stopping and best-parameter restoration. Audit is reserved.

Outputs: `outputs/4b/trained/{calibrator.pt,pca.pkl,training.json,progress.jsonl}`.

### Step 3: Test and deploy the new checkpoint

```bash
python primaconsis.py evaluate --config config.yaml --split test --trained
```

To deploy it online, set `paths.calibrator: outputs/4b/trained` and instantiate `CalibratedJev` again.

---

## Paper Results

### Native JEV: calibration-set consistency (Tables 1–4)

**Axis 1.**

| Model | G(n+, n−) ↓ | G(n+, c) ↓ | G(n−, c) ↓ |
|---|---:|---:|---:|
| JevK5-2B | 0.1858 | 0.2471 | 0.1127 |
| JevK5-4B | 0.1360 | 0.1132 | 0.0689 |
| JevK5-9B | 0.0892 | 0.0972 | 0.0686 |

**Axis 2.**

| Model | Coarse TV ↓ | Medium TV ↓ | Fine TV ↓ |
|---|---:|---:|---:|
| JevK5-2B | 0.2945 | 0.4837 | 0.4890 |
| JevK5-4B | 0.0870 | 0.5329 | 0.6323 |
| JevK5-9B | 0.1300 | 0.3650 | 0.5107 |

**Axis 3.**

| Model | Abstention gap ↓ | Conflict (%) ↓ | Accuracy gap (pp) ↓ |
|---|---:|---:|---:|
| JevK5-2B | 0.1928 | 36.54 | 9.66 |
| JevK5-4B | 0.2614 | 26.69 | 10.83 |
| JevK5-9B | 0.2130 | 17.18 | 11.72 |

**Axis 4.**

| Model | TV(raw, desc) ↓ | TV(raw, Noul) ↓ | TV(desc, Noul) ↓ |
|---|---:|---:|---:|
| JevK5-2B | 0.0917 | 0.4386 | 0.4716 |
| JevK5-4B | 0.1013 | 0.2405 | 0.2694 |
| JevK5-9B | 0.0439 | 0.2227 | 0.2450 |

### CPRC: in-domain test results (Table 5)

**Axis I.**

| Model | G(n+, n−) ↓ | G(n+, c) ↓ | G(n−, c) ↓ | Accuracy (%) ↑ |
|---|---:|---:|---:|---:|
| JevK5-2B | 0.1832 | 0.2442 | 0.1106 | 65.03 |
| JevK5-2B + CPRC | 0.0869 | 0.0962 | 0.0643 | 79.00 |
| JevK5-4B | 0.1361 | 0.1132 | 0.0688 | 71.77 |
| JevK5-4B + CPRC | 0.0577 | 0.0619 | 0.0511 | 82.80 |
| JevK5-9B | 0.0861 | 0.0931 | 0.0654 | 71.22 |
| JevK5-9B + CPRC | 0.0567 | 0.0633 | 0.0575 | 80.98 |

**Axis II.**

| Model | Coarse TV ↓ | Medium TV ↓ | Fine TV ↓ | Accuracy (%) ↑ |
|---|---:|---:|---:|---:|
| JevK5-2B | 0.2923 | 0.4806 | 0.4867 | 77.67 |
| JevK5-2B + CPRC | 0.0666 | 0.1296 | 0.1573 | 98.48 |
| JevK5-4B | 0.0870 | 0.5330 | 0.6325 | 95.20 |
| JevK5-4B + CPRC | 0.0842 | 0.2815 | 0.3545 | 99.65 |
| JevK5-9B | 0.1292 | 0.3618 | 0.5085 | 93.77 |
| JevK5-9B + CPRC | 0.0890 | 0.1756 | 0.2534 | 99.13 |

**Axis III.**

| Model | Abstention gap ↓ | Conflict (%) ↓ | Accuracy gap (pp) ↓ | Accuracy (%) ↑ |
|---|---:|---:|---:|---:|
| JevK5-2B | 0.1884 | 36.15 | 8.77 | 77.81 |
| JevK5-2B + CPRC | 0.2260 | 18.37 | 6.92 | 87.96 |
| JevK5-4B | 0.2619 | 26.74 | 11.03 | 85.63 |
| JevK5-4B + CPRC | 0.2018 | 14.84 | 9.90 | 93.58 |
| JevK5-9B | 0.2103 | 17.27 | 11.51 | 91.79 |
| JevK5-9B + CPRC | 0.1802 | 12.83 | 9.73 | 95.67 |

### Held-out Axis IV (Tables 6–7)

| Model | TV(raw, desc) ↓ | TV(raw, Noul) ↓ | TV(desc, Noul) ↓ | Raw-label accuracy (%) ↑ | Described-label accuracy (%) ↑ |
|---|---:|---:|---:|---:|---:|
| JevK5-2B | 0.0825 | 0.4391 | 0.4736 | 84.90 | 86.80 |
| JevK5-2B + CPRC | 0.0988 | 0.1819 | 0.2369 | 87.42 | 87.90 |
| JevK5-4B | 0.0907 | 0.2407 | 0.2696 | 84.08 | 86.97 |
| JevK5-4B + CPRC | 0.0663 | 0.1719 | 0.2083 | 84.12 | 87.05 |
| JevK5-9B | 0.0447 | 0.2251 | 0.2477 | 88.02 | 89.03 |
| JevK5-9B + CPRC | 0.0513 | 0.0936 | 0.1262 | 89.75 | 90.07 |

---

## Acknowledgment

We thank the authors of [JevK5](https://github.com/allebee/jevk5), [HaluEval](https://github.com/RUCAIBox/HaluEval), [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), [AbstentionBench](https://huggingface.co/datasets/facebook/AbstentionBench), and [AG News](https://huggingface.co/datasets/fancyzhx/ag_news). Original source data terms continue to apply.

## Citation

```bibtex
@misc{liu2026crossprimitive,
  title={Cross-Primitive Consistency in Jev},
  author={Xu Liu and Guikun Chen and Kanzhi Wu and Wenguan Wang},
  year={2026},
  eprint={2610.38623},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.38623}
}
```
