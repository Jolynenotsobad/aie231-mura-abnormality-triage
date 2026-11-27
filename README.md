# MURA Abnormality Triage

**Binary classification of musculoskeletal X-ray abnormalities using deep learning.**

> AIE231 — Neural Networks Course Project  
> Alamein International University · Faculty of Computer Science & Engineering

---

## Problem Statement

A radiologist screening musculoskeletal X-rays needs automated triage assistance. This project builds a neural-network pipeline that classifies each radiograph as **Normal** or **Abnormal** (indicating a fracture, dislocation, or other pathology).

| Aspect | Detail |
|--------|--------|
| **Input** | 224 × 224 × 3 RGB X-ray image |
| **Output** | P(abnormal) ∈ [0, 1] via sigmoid |
| **Loss** | Binary cross-entropy with class weights |
| **Primary metric** | Cohen's Kappa (κ) — agreement adjusted for chance |
| **Secondary metrics** | Recall (sensitivity), Macro-F1, AUC |

**Why κ and recall?** A missed abnormality (false negative) delays patient treatment — the costlier error. Kappa accounts for the class imbalance (61 % normal / 39 % abnormal), and high recall ensures abnormalities are caught.

## Dataset

**MURA v1.1** — Musculoskeletal Radiograph Abnormality Detection  
Rajpurkar et al., 2017 · [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)

| Split | Studies | Images | Normal | Abnormal |
|-------|---------|--------|--------|----------|
| Train | 13,457 | 36,808 | 61.5 % | 38.5 % |
| Valid | 1,199 | 3,197 | 55.1 % | 44.9 % |

Seven upper-extremity body parts: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist.

## Results — Baselines (Checkpoint 2)

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | κ |
|-------|----------|-----------|--------|----|----------|---|
| Majority class | ~0.615 | 0.000 | 0.000 | 0.000 | — | 0.000 |
| Logistic regression | — | — | — | — | — | — |

> *Full results will be populated after running the baseline evaluation.*  
> *Neural network results (Models A–C) will be added in Checkpoint 3.*

## Repository Structure

```
aie231-mura-abnormality-triage/
├── README.md                  ← this file
├── requirements.txt           pinned package versions
├── configs/                   YAML experiment configurations
│   ├── base.yaml              shared defaults
│   ├── exp_baseline.yaml      baseline evaluation
│   ├── exp_custom_cnn.yaml    Model A — custom CNN
│   ├── exp_resnet50_frozen.yaml   Model B — frozen ResNet-50
│   └── exp_resnet50_finetune.yaml Model C — fine-tuned ResNet-50
├── data/
│   └── README.md              download instructions (no raw data in git)
├── notebooks/
│   ├── 01_eda.ipynb           exploratory data analysis
│   └── 02_results.ipynb       final results & visualisations
├── src/
│   ├── __init__.py
│   ├── data.py                loading, cleaning, splitting, preprocessing
│   ├── models.py              baselines and neural architectures
│   ├── train.py               training loop, callbacks, logging
│   ├── evaluate.py            metrics, confusion matrix, ROC, error analysis
│   └── predict.py             single-image inference pipeline
├── experiments/
│   └── experiment_log.csv     run log (CSV)
├── models/                    saved models (git-ignored)
├── app/
│   └── app.py                 Gradio demo application
└── report/                    technical report and slides
```

## Setup & Reproduction

### 1. Clone and install

```bash
git clone https://github.com/Jolynenotsobad/aie231-mura-abnormality-triage.git
cd aie231-mura-abnormality-triage
pip install -r requirements.txt
```

### 2. Download the dataset

Download [MURA v1.1](https://www.kaggle.com/datasets/cjinny/mura-v11) and extract it into the `data/` directory:

```
data/MURA-v1.1/
├── train/
├── valid/
├── train_image_paths.csv
├── train_labeled_studies.csv
├── valid_image_paths.csv
└── valid_labeled_studies.csv
```

### 3. Run the EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Run baseline evaluation

```bash
python -m src.train --config configs/exp_baseline.yaml --baseline
```

## Team & Roles

| Role | Work Package |
|------|-------------|
| 1. Tech Lead | Problem formulation, repo structure, integration, final model selection |
| 2. Data Engineer | Data loading, cleaning, splitting, augmentation |
| 3. Baseline & Analysis Engineer | EDA, baselines, Model A |
| 4. Model Architect | Architecture design, Models B & C, regularisation |
| 5. Experimentation Engineer | Hyperparameter tuning, experiment tracking |
| 6. Evaluation & Deployment Engineer | Test-set evaluation, error analysis, inference pipeline, demo |

## Citation

```bibtex
@article{rajpurkar2017mura,
  title={MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Bagul, Aarti and others},
  journal={arXiv preprint arXiv:1712.06957},
  year={2017}
}
```

## Ethical Considerations

This model is a **course project** and is **not a diagnostic tool**. It should not be used for clinical decision-making. The MURA dataset is de-identified and released for academic research under a Stanford University research use agreement.

## Licence

This project is for academic use only as part of the AIE231 course.
