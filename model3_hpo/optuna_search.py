# Bu dosyada Optuna ile hiperparametre aramasi yapacagiz.
# Amac: learning_rate, batch_size ve optimizer icin en iyi kombinasyonu bulmak.

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from data_prep import get_dataloaders
from model import CatVsJetCNN

# Her trial sonucunu saklamak icin liste
trial_results = []

def objective(trial):
    # Optuna'nin deneyecegi hiperparametreler
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    # Search asamasi icin epoch dusuk tutulur (hizli deneme icin)
    epochs = 3

    # Donanim secimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader'lari al
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Model + loss
    model = CatVsJetCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer secimi
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Egitim
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation olcumu
    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    val_loss = val_running_loss / len(val_loader)

    # Trial sonuclarini kaydet (rapor icin)
    trial_results.append({
        "trial_number": trial.number,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "epochs": epochs,
        "val_acc": val_acc,
        "val_loss": val_loss
    })

    # Optuna objective degeri (maximize edecegiz)
    return val_acc

def main():
    # Sonuc klasoru
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    trials_csv_path = os.path.join(results_dir, "optuna_trials_model3_hpo.csv")
    best_params_path = os.path.join(results_dir, "optuna_best_params_model3_hpo.txt")

    # Study olustur (amac: val_acc max)
    study = optuna.create_study(direction="maximize")

    # Trial sayisi
    n_trials = 20

    print("Optuna search basliyor...\n")
    print(f"Toplam trial sayisi: {n_trials}\n")

    study.optimize(objective, n_trials=n_trials)

    # Tum trial'lari CSV'ye yaz
    with open(trials_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_number", "learning_rate", "batch_size", "optimizer", "epochs", "val_acc", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(trial_results)

    # En iyi sonucu yaz
    best_trial = study.best_trial
    with open(best_params_path, mode="w", encoding="utf-8") as f:
        f.write("Model 3 - Optuna Best Params\n")
        f.write("----------------------------\n")
        f.write(f"best_trial_number: {best_trial.number}\n")
        f.write(f"best_val_acc: {best_trial.value:.2f}%\n")
        f.write(f"learning_rate: {best_trial.params['learning_rate']}\n")
        f.write(f"batch_size: {best_trial.params['batch_size']}\n")
        f.write(f"optimizer: {best_trial.params['optimizer']}\n")
        f.write("search_epochs_per_trial: 3\n")
        f.write("final_training_epochs: 5\n")

    print("Optuna search tamamlandi.")
    print(f"Trial sonuclari: {trials_csv_path}")
    print(f"En iyi parametreler: {best_params_path}")

if __name__ == "__main__":
    main()