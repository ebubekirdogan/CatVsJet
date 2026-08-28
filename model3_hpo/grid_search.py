# Bu dosyada kucuk bir Grid Search yapacagiz.
# Amac: learning_rate ve batch_size kombinasyonlarini deneyip
# en iyi validation accuracy veren ayarlari bulmak.

import os
import csv
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

from data_prep import get_dataloaders
from model import CatVsJetCNN

def run_one_experiment(learning_rate, batch_size, epochs=3, optimizer_name="adam"):
    # Donanim secimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader'lari al
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Model + loss
    model = CatVsJetCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer secimi
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("optimizer_name sadece 'adam' veya 'sgd' olabilir.")

    # Egitim dongusu
    for epoch in range(epochs):
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

    return val_acc, val_loss

def main():
    # ============================================
    # Grid Search uzayi (kucuk ve anlasilir)
    # ============================================
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [16, 32, 64]
    epochs_for_search = 3
    optimizer_name = "adam"
    # ============================================

    # Sonuc klasoru
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_csv_path = os.path.join(results_dir, "grid_search_results_model3_hpo.csv")
    best_params_path = os.path.join(results_dir, "best_grid_params_model3_hpo.txt")

    # Tum kombinasyonlari olustur
    all_combinations = list(itertools.product(learning_rates, batch_sizes))

    print("Grid Search basliyor...\n")
    print(f"Toplam deneme sayisi: {len(all_combinations)}\n")

    all_results = []
    best_result = None  # en iyi sonucu burada tutacagiz

    for idx, (lr, bs) in enumerate(all_combinations, start=1):
        print(f"Deneme {idx}/{len(all_combinations)} | lr={lr} | batch_size={bs}")

        val_acc, val_loss = run_one_experiment(
            learning_rate=lr,
            batch_size=bs,
            epochs=epochs_for_search,
            optimizer_name=optimizer_name
        )

        print(f"--> Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}\n")

        row = {
            "learning_rate": lr,
            "batch_size": bs,
            "optimizer": optimizer_name,
            "epochs": epochs_for_search,
            "val_acc": val_acc,
            "val_loss": val_loss
        }
        all_results.append(row)

        # En iyi sonucu secme (once val_acc yuksek olsun, esitlikte val_loss dusuk olsun)
        if best_result is None:
            best_result = row
        else:
            if row["val_acc"] > best_result["val_acc"]:
                best_result = row
            elif row["val_acc"] == best_result["val_acc"] and row["val_loss"] < best_result["val_loss"]:
                best_result = row

    # CSV'ye yaz
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["learning_rate", "batch_size", "optimizer", "epochs", "val_acc", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # En iyi parametreyi txt'ye yaz
    with open(best_params_path, mode="w", encoding="utf-8") as f:
        f.write("Model 3 - Grid Search Best Params\n")
        f.write("----------------------------------\n")
        f.write(f"learning_rate: {best_result['learning_rate']}\n")
        f.write(f"batch_size: {best_result['batch_size']}\n")
        f.write(f"optimizer: {best_result['optimizer']}\n")
        f.write(f"epochs(search): {best_result['epochs']}\n")
        f.write(f"best_val_acc: {best_result['val_acc']:.2f}%\n")
        f.write(f"best_val_loss: {best_result['val_loss']:.4f}\n")

    print("Grid Search tamamlandi.")
    print(f"Sonuclar: {results_csv_path}")
    print(f"En iyi parametre: {best_params_path}")

if __name__ == "__main__":
    main()