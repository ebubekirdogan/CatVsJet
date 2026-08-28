# Bu dosyada egitilen modeli test verisi uzerinde degerlendirecegiz.
# Ayrica history dosyasindan train/val grafiklerini cizecegiz.
# Sonuclar results/ klasorune kaydedilecek.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from data_prep import get_dataloaders
from model import CatVsJetCNN


def evaluate_model(
    model_path="cat_vs_jet_model3_hpo.pth",
    history_path="history_model3_hpo.pth",
    results_dir="results",
    batch_size=32
):
    # Sonuc klasoru yoksa olustur
    os.makedirs(results_dir, exist_ok=True)

    # Donanim secimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Degerlendirme donanimi: {device.type.upper()}")

    # Dataloader
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # Modeli olusturup agirliklari yukle
    model = CatVsJetCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Gercek ve tahmin etiketlerini burada tutacagiz
    all_labels = []
    all_preds = []

    # Test tahminleri
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrikler
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Sinif isimleri (dataset/train klasor isimlerinden otomatik gelir)
    train_loader, _, _ = get_dataloaders(batch_size=batch_size)
    class_names = train_loader.dataset.classes

    # Metrikleri txt dosyasina yaz
    metrics_path = os.path.join(results_dir, "metrics_model3_hpo.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Model 3 (HPO) Test Metrics\n")
        f.write("--------------------------\n")
        f.write(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Confusion matrix gorseli
    cm_fig_path = os.path.join(results_dir, "confusion_matrix_model3_hpo.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Model 3 (HPO) - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()

    # History dosyasini yukle ve train/val grafiklerini ciz
    history = torch.load(history_path, map_location="cpu", weights_only=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(12, 5))

    # Loss grafigi
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.title("Model 3 (HPO) - Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # Accuracy grafigi
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="Val Acc")
    plt.title("Model 3 (HPO) - Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    curves_path = os.path.join(results_dir, "loss_accuracy_curves_model3_hpo.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=200)
    plt.close()

    print("Degerlendirme tamamlandi.")
    print(f"Metrikler: {metrics_path}")
    print(f"Confusion Matrix: {cm_fig_path}")
    print(f"Loss/Accuracy Grafik: {curves_path}")


if __name__ == "__main__":
    evaluate_model()