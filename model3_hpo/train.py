# Modelin egitildigi ana dosya. Bu dosya calistirildiginda model egitimi baslar ve egitim sonunda modelin agirliklari kaydedilir.
import torch
import torch.nn as nn
import torch.optim as optim # optimizer'larin bulundugu kutuphane (Adam, SGD gibi)
import argparse

from data_prep import get_dataloaders
from model import CatVsJetCNN

def train_model(
    learning_rate=0.001,
    batch_size=32,
    epochs=5,
    optimizer_name="adam",
    save_model_path='cat_vs_jet_model3_hpo.pth',
    save_history_path='history_model3_hpo.pth'
):
    # PyTorch'a CUDA (NVIDIA) var mı diye soruyoruz, varsa gücü oraya aktarıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan Donanim Motoru: {device.type.upper()}")

    # ============================================
    # 3 deger donuyor (train, val, test)
    # ============================================
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    # egitim paketi train_loadera, dogrulama paketi val_loadera, test paketi test_loadera ataniyor.

    # Sinifdan model nesnesini olusturuyoruz. bu cagrildigi anda __init__ fonksiyonu calisir ve modelin katmanlari olusturulur. model nesnesi ile artik ileri ve geri yayilim islemleri yapilabilir.
    model = CatVsJetCNN().to(device) # to device ile model nesnesini CUDA'ya (GPU) tasiyoruz. eger GPU yoksa CPU da calisir.

    # Hata fonksiyonu (Loss Function) : modelin sonucu ile gercek sonuc karsilastirir (cross-entropy yontemi ile : 0 mı 1 mi problemlerinde kulanilir.)
    criterion = nn.CrossEntropyLoss()

    # optimizer secimi
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("optimizer_name sadece 'adam' veya 'sgd' olabilir.")

    # ============================================
    # gecmisi kaydetmek icin listeler
    # ============================================
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    # ============================================

    print("Eğitim basliyor...\n")

    for epoch in range(epochs): # dis dongu; her epoch icin
        # ============================================
        # EĞİTİM (TRAIN) AŞAMASI
        # ============================================
        model.train() # model egitim moduna alinir. agirliklar güncellenecek.
        running_loss = 0.0 # hata sayaci. her epoch icin sifirla.
        train_correct = 0 # dogru bilinen resim sayaci
        train_total = 0 # toplam resim sayaci

        # enumerate kullanarak kaçıncı pakette olduğumuzu (i) sayıyoruz
        for i, (images, labels) in enumerate(train_loader):
            # train_loader dan batchler halinde resim ve etiketleri aliyoruz.
            images, labels = images.to(device), labels.to(device)

            # 1) onceki adimlardan hata copleri varsa bunları sifirla.
            optimizer.zero_grad()

            # 2) ileri yayılım. Resimleri modele ver. Tahminleri al.
            outputs = model(images)

            # 3) Hata hesapla. Tahminler ile gercek etiketleri karsilastir.
            loss = criterion(outputs, labels)

            # 4) Geriye Yayılım (Backward). Hatayı geriye doğru gönder, türevleri hesapla.
            loss.backward()

            # 5) Agirliklari guncelle.
            optimizer.step()

            # Ekrana bilgi basmak icin hatayi sayacimiza ekle.
            running_loss += loss.item()

            # train dogruluğunu da anlik olarak hesapliyoruz
            _, predicted = torch.max(outputs, 1) # en yuksek skora sahip sinifi sec
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if i == 0:
                print(f"\n===> Epoch {epoch+1}/{epochs} başladı ({len(train_loader)} paket işlenecek)")
            # Her 10 pakette bir anlık durumu yazdır
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Paket [{i+1}/{len(train_loader)}] | Anlık Hata: {loss.item():.4f}")

        # o epoch'un ortalama train loss ve accuracy'sini hesapla
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # ============================================
        # VALIDATION AŞAMASI
        # ============================================
        model.eval() # model DEGERLENDIRME moduna alinir.

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # validation'da OGRENME yapmiyoruz, sadece olcum yapiyoruz.
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images) # ileri yayilim (sadece tahmin, geri yayilim YOK)
                loss = criterion(outputs, labels) # hata hesapla (ama loss.backward() YAPMIYORUZ)

                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        # ============================================

        # gecmis listelerine bu epoch'un sonuclarini ekle
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Her turun (epoch) sonunda train VE val sonuclarini yan yana ekrana bas.
        print(
            f'===> Epoch [{epoch + 1}/{epochs}] TAMAMLANDI | '
            f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%\n'
        )

    print("Eğitim tamamlandı.")

    # Modeli kaydetme
    torch.save(model.state_dict(), save_model_path)
    print(f"Model basariyla {save_model_path} adiyla kaydedildi.")

    # history sozlugunu de kaydediyoruz
    torch.save(history, save_history_path)
    print(f"Egitim gecmisi (history) basariyla {save_history_path} adiyla kaydedildi.")

    return history


if __name__ == "__main__":
    # Komut satirindan parametre almak icin parser
    parser = argparse.ArgumentParser(description="Model 3 egitim dosyasi")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Ogrenme orani (lr)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch sayisi")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer secimi")
    parser.add_argument("--save-model-path", type=str, default="cat_vs_jet_model3_hpo.pth", help="Model kayit yolu")
    parser.add_argument("--save-history-path", type=str, default="history_model3_hpo.pth", help="History kayit yolu")

    args = parser.parse_args()

    train_model(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        save_model_path=args.save_model_path,
        save_history_path=args.save_history_path
    )