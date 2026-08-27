# Modelin egitildigi ana dosya. Bu dosya calistirildiginda model egitimi baslar ve egitim sonunda modelin agirliklari kaydedilir.
import torch
import torch.nn as nn
import torch.optim as optim # optimizer'larin bulundugu kutuphane (Adam, SGD gibi)

from data_prep import get_dataloaders
from model import CatVsJetCNN

def train_model():
    # PyTorch'a CUDA (NVIDIA) var mı diye soruyoruz, varsa gücü oraya aktarıyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan Donanim Motoru: {device.type.upper()}")

    train_loader, test_loader = get_dataloaders(batch_size=32) #get_dataloaders; data_prep.pydeki fonksiyonu calistirip 32 lik gruplar halinde hazır hale getirir.
    # egitim paketi train_loadera, test paketi test_loadera ataniyor.

    # Sinifdan model nesnesini olusturuyoruz. bu cagrildigi anda __init__ fonksiyonu calisir ve modelin katmanlari olusturulur. model nesnesi ile artik ileri ve geri yayilim islemleri yapilabilir.
    model = CatVsJetCNN().to(device) # model nesnesini CUDA'ya (GPU) tasiyoruz. eger GPU yoksa CPU da calisir.

    # Hata fonksiyonu (Loss Function) : modelin sonucu ile gercek sonuc karsilastirir (cross-entropy yontemi ile : 0 mı 1 mi problemlerinde kulanilir.)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001) # modelin parametrelerini optimize eden optimizer (Adam yontemi ile. ogrenme oranı = 0.001)
    # ayrica model.parameters() ile modelin ogrenilebilir agirliklarini optimizer'a veriyoruz ki optimizer agirliklari guncelleyebilsin.

    epochs = 5 # tum veri seti toplamda 5 kez okunacak.

    print("Eğitim basliyor...\n")

    for epoch in range(epochs): # dis dongu; her epoch icin
        model.train() # model egitim moduna alinir. agirliklar güncellenecek.
        running_loss = 0.0 # hata sayaci. her epoch icin sifirla.

        # YENİ HALİ: enumerate kullanarak kaçıncı pakette olduğumuzu (i) sayıyoruz
        for i, (images, labels) in enumerate(train_loader):
            # train_loader dan 32 li batchler halinde resim ve etiketleri aliyoruz.
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

            if i == 0:
                print(f"---> İLK PAKET (32 RESİM) GPU'YA GİRDİ VE ISLENDİ.")
            # Her 10 pakette bir anlık durumu yazdır
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Paket [{i+1}/{len(train_loader)}] | Anlık Hata: {loss.item():.4f}")
                
        # Her turun (epoch) sonunda ortalama hatayı ekrana bas.
        # toplam hatayı, kaç tane batch olduğuna (len(train_loader), yani train_loader'ın kaç batch ürettiği) bölüyoruz — bu bize o epoch'taki ortalama hatayı veriyor.
        print(f'===> Epoch [{epoch + 1}/{epochs}] TAMAMLANDI, Ortalama Loss: {running_loss / len(train_loader):.4f}\n')

    print("Eğitim tamamlandı.")
    
    # Modeli kaydetme
    # Modeli kaydetme amacimiz : Eğitim (özellikle büyük veri setlerinde) saatler sürebilir. Ağırlıkları kaydedersek, bir dahaki sefere modeli sıfırdan eğitmek yerine model.load_state_dict(torch.load('cat_vs_jet_model.pth')) diyerek kaldığımız yerden (ya da doğrudan tahmin yapmak için) devam edebiliriz.
    # model.state_dict() → modelin tüm öğrenilmiş ağırlıklarını (conv1, conv2, fc1, fc2'nin son haldeki sayılarını) bir Python sözlüğü (dictionary) olarak döndürür. Mimarinin kendisini değil, sadece o mimarideki sayıları içerir.
    # torch.save(..., 'cat_vs_jet_model.pth') → bu sözlüğü diske, .pth uzantılı bir dosyaya kaydediyor. .pth, PyTorch'un ağırlık dosyaları için yaygın kullanılan uzantısı (aslında içerik olarak sadece serileştirilmiş bir Python objesi).
    torch.save(model.state_dict(), 'cat_vs_jet_model.pth') # ogrenilmis tum agirliklari dosyaya kaydediyoruz ki her seferinde bastan egitmeyelim.
    print("Model basariyla cat_vs_jet_model.pth adiyla kaydedildi.")

if __name__ == "__main__":
    train_model()