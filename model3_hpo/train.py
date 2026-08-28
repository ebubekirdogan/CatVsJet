# Modelin egitildigi ana dosya. Bu dosya calistirildiginda model egitimi baslar ve egitim sonunda modelin agirliklari kaydedilir.
import torch
import torch.nn as nn
import torch.optim as optim # optimizer'larin bulundugu kutuphane (Adam, SGD gibi)

from data_prep import get_dataloaders
from model import CatVsJetCNN

# learning_rate, batch_size ve epochs artik fonksiyonun icine sabit yazilmiyor, disaridan parametre
# olarak aliniyor. Sebebi: Model 3'te Optuna bu fonksiyonu onlarca kere, her seferinde FARKLI
# learning_rate/batch_size degerleriyle cagiracak 
#
# save_files: Optuna arama yaparken onlarca kisa deneme yapacak, bunlarin agirligini diske
# kaydetmeye gerek yok (gereksiz yer kaplar, yavaslatir). Sadece en sonunda, en iyi
# hiperparametrelerle yapilacak asil egitimde save_files=True verecegiz.
#
# verbose: Arama sirasinda onlarca deneme oldugu icin her birinin pakete pakete hatasini
# ekrana basmak istemiyoruz, kalabalik olur. Sadece final egitimde verbose=True birakacagiz.
def train_model(learning_rate=0.001, batch_size=32, epochs=5, save_files=True, verbose=True):
    # PyTorch'a CUDA (NVIDIA) var mı diye soruyoruz, varsa gücü oraya aktarıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # ogrenme orani artik sabit degil, disaridan gelen learning_rate
    # ayrica model.parameters() ile modelin ogrenilebilir agirliklarini optimizer'a veriyoruz ki optimizer agirliklari guncelleyebilsin.

    # epochs artik disaridan parametre olarak geliyor, burada sabit yazmiyorum.

    # ============================================
    #  gecmisi kaydetmek icin listeler
    # ============================================
    # Bu listeler her epoch sonunda doldurulacak. Sonrasinda (baska bir dosyada)
    #***Bu listeleri kullanarak train ve validation icin "loss/accuracy" grafigini cizecegiz.***#
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    # ============================================

    if verbose:
        print(f"Eğitim basliyor... (lr={learning_rate}, batch_size={batch_size}, epochs={epochs})\n")

    for epoch in range(epochs): # dis dongu; her epoch icin
        # ============================================
        # EĞİTİM (TRAIN) AŞAMASI -
        # ============================================
        model.train() # model egitim moduna alinir. agirliklar güncellenecek.
        running_loss = 0.0 # hata sayaci. her epoch icin sifirla.
        train_correct = 0 # dogru bilinen resim sayaci 
        train_total = 0 # toplam resim sayaci 

        #  enumerate kullanarak kaçıncı pakette olduğumuzu (i) sayıyoruz
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
            # Nerede hata yaptık, hangi ağırlığı ne kadar değiştirmeliyiz hesapla.
            loss.backward()

            # 5) Agirliklari guncelle.
            optimizer.step()

            # Ekrana bilgi basmak icin hatayi sayacimiza ekle.
            running_loss += loss.item()

            # train dogruluğunu da anlik olarak hesapliyoruz
            _, predicted = torch.max(outputs, 1) # en yuksek skora sahip sinifi sec
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # arama sirasinda onlarca deneme oldugu icin bu ekran ciktilarini verbose ile kapatiyorum
            if verbose and i == 0:
                print(f"\n===> Epoch {epoch+1}/{epochs} başladı ({len(train_loader)} paket işlenecek)")
            # Her 10 pakette bir anlık durumu yazdır
            if verbose and (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Paket [{i+1}/{len(train_loader)}] | Anlık Hata: {loss.item():.4f}")

        # o epoch'un ortalama train loss ve accuracy'sini hesapla
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # ============================================
        # VALIDATION
        # ============================================
        # Bu blok her epoch'un sonunda, egitim bittikten hemen sonra calisir.
        # Amac: model bu epoch sonunda hic gormedigi (val_loader) veride ne kadar basarili, onu olcmek.
        model.eval() # model DEGERLENDIRME moduna alinir.

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # torch.no_grad() -> bu blok icinde turev (gradient) hesaplanmaz.
        # Cunku validation'da OGRENME yapmiyoruz, sadece olcum yapiyoruz. Bu ayni zamanda islemi hizlandirir.
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images) # ileri yayilim (sadece tahmin, geri yayilim YOK)
                loss = criterion(outputs, labels) # hata hesapla (ama loss.backward() YAPMIYORUZ, agirlik guncellemiyoruz)

                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        # ============================================

        # gecmis listelerine bu epoch'un sonuclarini ekle (grafik cizerken kullanacagiz)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Her turun (epoch) sonunda train VE val sonuclarini yan yana ekrana bas.
        if verbose:
            print(f'===> Epoch [{epoch + 1}/{epochs}] TAMAMLANDI | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%\n')

    if verbose:
        print("Eğitim tamamlandı.")

    # Modeli kaydetme
    # Modeli kaydetme amacimiz : Eğitim (özellikle büyük veri setlerinde) saatler sürebilir. Ağırlıkları kaydedersek, bir dahaki sefere modeli sıfırdan eğitmek yerine model.load_state_dict(torch.load('cat_vs_jet_model.pth')) diyerek kaldığımız yerden (ya da doğrudan tahmin yapmak için) devam edebiliriz.
    # model.state_dict() → modelin tüm öğrenilmiş ağırlıklarını (conv1, conv2, fc1, fc2'nin son haldeki sayılarını) bir Python sözlüğü (dictionary) olarak döndürür. Mimarinin kendisini değil, sadece o mimarideki sayıları içerir.
    # torch.save(..., 'cat_vs_jet_model.pth') → bu sözlüğü diske, .pth uzantılı bir dosyaya kaydediyor. .pth, PyTorch'un ağırlık dosyaları için yaygın kullanılan uzantısı (aslında içerik olarak sadece serileştirilmiş bir Python objesi).
    #
    # save_files=False iken (Optuna arama sirasinda) bu blok hic calismiyor, cunku her denemenin
    # agirligini diske yazmaya gerek yok, sadece final egitimde (save_files=True) kaydediyoruz.
    if save_files:
        torch.save(model.state_dict(), 'cat_vs_jet_model3_hpo.pth') # ogrenilmis tum agirliklari dosyaya kaydediyoruz ki her seferinde bastan egitmeyelim.
        if verbose:
            print("Model basariyla cat_vs_jet_model3_hpo.pth adiyla kaydedildi.")

        # history sozlugunu de kaydediyoruz. Bunu baska bir dosyada (grafik cizerken) kullanacagiz.
        torch.save(history, 'history_model3_hpo.pth')
        if verbose:
            print("Egitim gecmisi (history) basariyla history_model3_hpo.pth adiyla kaydedildi.")

    # Optuna, bir denemenin ne kadar iyi oldugunu anlamak icin tek bir sayiya ihtiyac duyuyor.
    # Bu yuzden history'nin yaninda, son epoch'un val accuracy'sini de ayrica donduruyoruz.
    final_val_acc = history["val_acc"][-1]
    return history, final_val_acc

if __name__ == "__main__":
    # dosya dogrudan calistirilirsa varsayilan degerlerle (lr=0.001, batch_size=32, epochs=5) egitir
    train_model()