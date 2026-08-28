# CatVsJet — 3 Model Comparison

This report compares the performance of the CatVsJet CNN model resulting from 3 different development steps. Each model uses the same base architecture (a simple CNN with 2 convolutional layers) and the same, fixed train/validation/test split (~78%/11%/11%, class-balanced). Each was trained for 5 epochs.

- **Model 1 (Baseline):** The base model with train/validation/test splitting and performance metrics added.
- **Model 2 (Augmented):** Built upon Model 1, with data augmentation (horizontal flip, brightness/contrast changes, ±15° rotation) added only to the training data.
- **Model 3 (HPO):** Built upon Model 1, with the learning_rate and batch_size hyperparameters optimized using Optuna (no augmentation).

Model 2 and Model 3 were created independently based on Model 1 — meaning both were developed from the same source (baseline), not on top of each other.

## Results Table (Test Set, 1184 images)

| Metric | Model 1 (Baseline) | Model 2 (Augmented) | Model 3 (HPO) |
|---|---|---|---|
| Accuracy | 91.72% | **92.99%** | 92.91% |
| Precision | 89.19% | **94.88%** | 93.78% |
| Recall | **94.92%** | 90.86% | 91.88% |
| F1-Score | 91.97% | **92.83%** | 92.82% |

## Confusion Matrices

| Model 1 | Model 2 | Model 3 |
|---|---|---|
| ![Model 1 CM](model1_baseline/results/confusion_matrix_model1_baseline.png) | ![Model 2 CM](model2_augmented/results/confusion_matrix_model2_augmented.png) | ![Model 3 CM](model3_hpo/results/confusion_matrix_model3_hpo.png) |

## Train vs Validation Graphs

| Model 1 | Model 2 | Model 3 |
|---|---|---|
| ![Model 1 curves](model1_baseline/results/loss_accuracy_curves_model1_baseline.png) | ![Model 2 curves](model2_augmented/results/loss_accuracy_curves_model2_augmented.png) | ![Model 3 curves](model3_hpo/results/loss_accuracy_curves_model3_hpo.png) |

## Comments

**Model 1 (Baseline):** At the 5th epoch, the train accuracy (95.32%) has surpassed the val accuracy (92.74%) — there is a gap of ~2.6 points between them. This is a sign that slight overfitting has begun. It is not a serious issue, but a trend that could grow with longer training.

**Model 2 (Augmented):** After data augmentation, the train and val accuracy have become very close to each other, and val accuracy even trended above train accuracy at times. This shows that augmentation effectively reduced overfitting — since the training data was presented in different ways (rotated, brightness changed) at each epoch, the model has now learned more general patterns instead of memorizing. Precision increased significantly (89.19% → 94.88%), while recall decreased (94.92% → 90.86%) — the model has become more selective/cautious when predicting "cat".

**Model 3 (HPO):** The model trained with `learning_rate ≈ 0.000376` and `batch_size = 16` found by Optuna showed a performance very close to Model 2 (an accuracy difference of only 0.08 points, and an F1 difference of 0.01 points). It provided a similar-sized gain through a different mechanism (by improving the learning process itself).

**Overall assessment:** Both data augmentation and hyperparameter optimization provided similarly sized improvements over the baseline, but in different ways — augmentation by reducing overfitting, and HPO by improving the learning process. The fact that the two turned out to be so close suggests that the current simple architecture (2 convolutional layers) might be approaching a performance ceiling for this dataset. A possible next step to try in the future: combining augmentation and HPO (using Model 2's data augmentation alongside the hyperparameters found by Optuna simultaneously), or adding layers to the architecture.
