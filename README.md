<div align="center">

# CatVsJet

**A from-scratch PyTorch CNN for cat vs. jet image classification.**
Three isolated experiments — baseline, data augmentation, and Optuna-based hyperparameter optimization — benchmarked side by side.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-HPO-3670A0)
![matplotlib](https://img.shields.io/badge/matplotlib-visualization-11557C)

📊 [**View the full comparison report →**](./comparison.md)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [3 Models, 3 Approaches](#-3-models-3-approaches)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)

---

## 🎯 About the Project

CatVsJet is a binary image classifier that distinguishes cats from jets (airplanes), built with a Convolutional Neural Network (CNN) written entirely from scratch — no pretrained models involved.

The goal of this project goes beyond just training a working model: it's about measuring performance correctly, isolating the true effect of data augmentation, and evaluating the contribution of hyperparameter optimization to the training process. For that reason, the project is structured as **3 independent experiments** run on the exact same architecture and the exact same data split, rather than a single model.

## 🧠 Model Architecture

The model is a simple but effective 2-layer convolutional CNN:

```
Input (3×64×64 RGB image)
   │
   ▼
Conv2D(3 → 16, kernel 3×3, padding 1) → ReLU → MaxPool(2×2)     # 16×32×32
   │
   ▼
Conv2D(16 → 32, kernel 3×3, padding 1) → ReLU → MaxPool(2×2)    # 32×16×16
   │
   ▼
Flatten (8192)
   │
   ▼
Linear(8192 → 512) → ReLU
   │
   ▼
Linear(512 → 2)   →  [airplane, cat] scores
```

All three models use this **exact same architecture** — the improvements happen in the training process (data augmentation, hyperparameters), not in the architecture itself.

## 🗂 Dataset

| | |
|---|---|
| **Classes** | `airplane`, `cat` |
| **Total images** | 10,705 (5,352 airplane + 5,353 cat, class-balanced) |
| **Split** | ~78% train / ~11% validation / ~11% test |
| **Preprocessing** | Resized to 64×64, converted to tensor, normalized |

The train/validation/test split preserves a near-equal class ratio across all three sets (class-balanced) and is fixed across all three models, making the comparison fair and consistent.

## 🔬 3 Models, 3 Approaches

| | Model 1 — Baseline | Model 2 — Augmented | Model 3 — HPO |
|---|---|---|---|
| **Architecture** | Standard CNN | Standard CNN (unchanged) | Standard CNN (unchanged) |
| **Data augmentation** | ❌ | ✅ Horizontal flip, brightness/contrast jitter (±30%), ±15° rotation *(training data only)* | ❌ |
| **Hyperparameters** | lr=0.001, batch=32 (default) | lr=0.001, batch=32 (default) | Optimized with Optuna: **lr≈0.000376, batch=16** |
| **Epochs** | 5 | 5 | 5 |
| **Goal** | Reference/comparison point | Reduce overfitting | Optimize the training process |

Model 2 and Model 3 were each built independently from Model 1 (not cumulatively) — so the effect of each technique can be measured in isolation, without one interfering with the other.

Model 3's hyperparameters were found through a 15-trial Optuna search (each trial a fast 2-epoch run); once the best combination was found, the final model was trained for the full 5 epochs.

## 📁 Project Structure

```
CatVsJet/
├── dataset/                       # train / val / test folders (with airplane, cat subfolders)
│   ├── train/{airplane,cat}
│   ├── val/{airplane,cat}
│   └── test/{airplane,cat}
│
├── model1_baseline/
│   ├── data_prep.py               # data loading (no augmentation)
│   ├── model.py                   # CNN architecture
│   ├── train.py                   # training (train + validation loop, history tracking)
│   ├── test.py                    # quick test (overall accuracy)
│   ├── evaluate.py                # precision/recall/f1/confusion matrix + plots
│   └── results/                   # confusion matrix, loss/accuracy curves, metrics report
│
├── model2_augmented/
│   ├── data_prep.py               # augmentation applied to TRAINING data only
│   ├── model.py, train.py, test.py, evaluate.py
│   └── results/
│
├── model3_hpo/
│   ├── data_prep.py, model.py     # identical to baseline
│   ├── train.py                   # learning_rate/batch_size/epochs as function arguments
│   ├── optuna_search.py           # Optuna hyperparameter search (15 trials)
│   ├── train_final.py             # trains the final model with the best hyperparameters found
│   ├── test.py, evaluate.py
│   └── results/                   # plots, metrics, optuna_best_params.txt
│
├── comparison.md                  # detailed 3-model comparison report
└── README.md
```

> `.pth` model weight files are excluded from the repo via `.gitignore` — they're large binary files that can't be rendered on GitHub anyway. Results are fully tracked through the plots in each `results/` folder and in `comparison.md`.

## ⚙️ Setup

```bash
git clone https://github.com/ebubekirdogan/CatVsJet.git
cd CatVsJet

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install torch torchvision scikit-learn matplotlib optuna
```

> PyTorch automatically uses a CUDA-enabled NVIDIA GPU if available, otherwise it runs on CPU.

> The `dataset/` folder (train/val/test, ~10.7k images) is included directly in this repository — no separate download step is needed. Once you clone the repo, the dataset is already in place.

## 🚀 Usage

Each model folder is self-contained — step into whichever one you want to explore and run the commands below.

### Model 1 — Baseline

```bash
cd model1_baseline
python train.py       # trains the model, saves weights (.pth) and training history
python evaluate.py    # computes metrics on the test set, saves plots to results/
```

### Model 2 — Augmented

```bash
cd model2_augmented
python train.py       # trains on augmented training data
python evaluate.py
```

### Model 3 — HPO (Optuna)

```bash
cd model3_hpo
python optuna_search.py   # searches for the best learning_rate / batch_size combo (~15 trials)
python train_final.py     # trains the final model for 5 epochs with the best hyperparameters found
python evaluate.py
```

## 📈 Results

Results on the held-out test set (1,184 images):

| Metric | Model 1 (Baseline) | Model 2 (Augmented) | Model 3 (HPO) |
|---|:---:|:---:|:---:|
| Accuracy | 91.72% | **92.99%** | 92.91% |
| Precision | 89.19% | **94.88%** | 93.78% |
| Recall | **94.92%** | 90.86% | 91.88% |
| F1-Score | 91.97% | **92.83%** | 92.82% |

**In short:** both data augmentation and hyperparameter optimization improved on the baseline by a similar margin — but through different mechanisms. Augmentation clearly reduced overfitting (the train/val gap closed), while hyperparameter optimization improved the efficiency of the learning process itself.

📊 For confusion matrices, training curves, and the full analysis, see **[comparison.md](./comparison.md)**.

## 🛠 Tech Stack

| Technology | Used for |
|---|---|
| **PyTorch / torchvision** | Model architecture, training loop, data loading and transforms |
| **scikit-learn** | Precision, recall, F1-score, and confusion matrix calculations |
| **matplotlib** | Loss/accuracy curves and confusion matrix visualization |
| **Optuna** | Hyperparameter optimization (learning rate, batch size search) |
