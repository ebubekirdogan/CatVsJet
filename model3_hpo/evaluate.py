# evaluate.py -> Egitilmis modelin TEST seti uzerindeki performansini olcen ve gorsellestiren dosya.
# Bu dosya train_final.py'den SONRA calistirilir (cunku .pth dosyalarina ihtiyac duyar).

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_prep import get_dataloaders
from model import CatVsJetCNN

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Degerlendirme Basliyor! Donanim: {device.type.upper()}\n")

    # sadece test_loader lazim, train ve val'i bu dosyada kullanmiyoruz
    _, _, test_loader = get_dataloaders(batch_size=16)

    # bos model olustur, egitilmis agirliklari yukle 
    model = CatVsJetCNN().to(device)
    model.load_state_dict(torch.load('cat_vs_jet_model3_hpo.pth', weights_only=True))
    model.eval()

    # ============================================
    # 1. ADIM: TEST SETINDE TAHMIN TOPLAMA
    # ============================================
    # sklearn fonksiyonlari batch batch degil, TUM test setinin tahmin/gercek listesini ister.
    # bu yuzden butun batch'lerdeki sonuclari iki BUYUK listede biriktiriyoruz.
    all_preds = []   # modelin tahminlerinin tamami buraya toplanacak
    all_labels = []  # gercek etiketlerin tamami buraya toplanacak

    with torch.no_grad():  # test'te ogrenme yok, turev hesaplamaya gerek yok
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                    # ileri yayilim, ham skorlar
            _, predicted = torch.max(outputs, 1)        # en yuksek skorlu sinifi sec

            # tensor'leri CPU'ya alip .numpy() ile Python listesine cevirip biriktiriyoruz
            # (sklearn PyTorch tensor degil, numpy array/Python listesi bekler)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ============================================
    # 2. ADIM: METRIKLERI HESAPLA (sklearn)
    # ============================================
    # her fonksiyona (gercek_etiketler, tahmin_edilenler) sirasiyla veriyoruz
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)  # 2x2 matris doner

    print("========== TEST SONUCLARI (Model 3 - HPO) ==========")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print("===========================================================")

    # sonuclari bir txt dosyasina da yaziyoruz, comparison.md yazarken buradan kopyalayacagiz
    with open('results/metrics_model3_hpo.txt', 'w', encoding='utf-8') as f:
        f.write("Model 3 - HPO Test Sonuclari\n")
        f.write(f"Accuracy  : {accuracy*100:.2f}%\n")
        f.write(f"Precision : {precision*100:.2f}%\n")
        f.write(f"Recall    : {recall*100:.2f}%\n")
        f.write(f"F1-Score  : {f1*100:.2f}%\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # ============================================
    # 3. ADIM: CONFUSION MATRIX GORSELI (heatmap)
    # ============================================
    class_names = ['airplane', 'cat']  # ImageFolder'in alfabetik sirada verdigi sinif isimleri

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')  # matrisi renkli kare olarak ciz, koyu = yuksek sayi
    plt.title('Confusion Matrix - Model 3 (HPO)')
    plt.colorbar()  # yandaki renk skalasi
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gercek')

    # her hucrenin icine sayiyi yaziyoruz (0,0 / 0,1 / 1,0 / 1,1 hucreleri)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig('results/confusion_matrix_model3_hpo.png')
    plt.close()  # bellekte acik kalmasin diye figuru kapatiyoruz

    # ============================================
    # 4. ADIM: LOSS / ACCURACY EGRILERI (history'den)
    # ============================================
    # train.py'nin kaydettigi history sozlugunu okuyoruz (train_loss, val_loss, train_acc, val_acc listeleri var)
    history = torch.load('history_model3_hpo.pth', weights_only=False)
    epochs_range = range(1, len(history["train_loss"]) + 1)  # x ekseni: 1, 2, 3, 4, 5 (epoch numaralari)

    plt.figure(figsize=(10, 4))

    # SOL grafik: Loss (Train vs Val)
    plt.subplot(1, 2, 1)  # 1 satir, 2 sutunluk grafik alaninin 1. si
    plt.plot(epochs_range, history["train_loss"], label='Train Loss')
    plt.plot(epochs_range, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()

    # SAG grafik: Accuracy (Train vs Val)
    plt.subplot(1, 2, 2)  # ayni grafik alaninin 2. si
    plt.plot(epochs_range, history["train_acc"], label='Train Acc')
    plt.plot(epochs_range, history["val_acc"], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Val Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/loss_accuracy_curves_model3_hpo.png')
    plt.close()

    print("\nGrafikler ve metrikler 'results/' klasorune kaydedildi.")

if __name__ == "__main__":
    evaluate_model()