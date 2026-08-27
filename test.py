import torch
from data_prep import get_dataloaders
from model import CatVsJetCNN

def test_model():
    # 1. DONANIM SEÇİMİ CUDA (NVIDIA) var mı diye soruyoruz, varsa gücü oraya aktarıyoruz. yoksa CPU da calisir.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sinav Başliyor! Donanim: {device.type.upper()}\n")

    # get_dataloaders iki deger donduruyordu: train_loader ve test_loader. ancak test dosyasında egitim verisine gerek yok. kullanmayacagiz.
    # '_' (alt çizgi) Python'da "bu veriyi kullanmayacağım, çöpe at" demektir.
    _, test_loader = get_dataloaders(batch_size=32)

    # 2. model olusturma calisinca model.pyde __init__ fonksiyonu calisir ve su katmanlar olusur: conv1, conv2, pool, fc1, fc2.
    model = CatVsJetCNN().to(device) # model nesnesini CUDA'ya (GPU) tasiyoruz. eger GPU yoksa CPU da calisir.
    # torch.load ile kaydedilen modelin ağırlıklarını yükleriz. weights_only=True ile sadece ağırlıkları yükleriz, mimariyi değil.
    # ve okunan agirliklar olusturulan model nesnesine atanir.
    model.load_state_dict(torch.load('cat_vs_jet_model.pth', weights_only=True))
    
    # Modeli test moduna alinir. 
    model.eval()

    # 3. istatistik tutacak sayaclar olusturulur.
    dogru_bilinen = 0
    toplam_resim = 0 # toplam test edilen resim sayisi.

    print("Model test ediliyor...")

    # 4. test döngüsü
    with torch.no_grad(): # burada türev hesaplamasi yapmayacagiz. modelin agirliklarini guncellemeyecegiz. sadece test edecegiz. bu yuzden torch.no_grad() ile geriye yayilim islemi kapatilir.
         for i, (images, labels) in enumerate(test_loader): #test_loader daki tum batchleri sira sira dolasir. batch batch
            # verileri ekran kartina tasiyoruz. images ve labels tensorleri GPU'ya tasinir. eger GPU yoksa CPU da calisir.
            images, labels = images.to(device), labels.to(device)
            # Resimleri modele ver, ham tahmin skorlarını al
            outputs = model(images) # resim batch i cnn modeline gonderilir. bu islem model.pydeki ileri yayilim fonk.nunu calistirir (forward function).
            # outputs [0.4,1.7] gibi 2 sayi cikar. bu sayilarin her biri bir sinifin skorunu temsil eder. 0: airplane, 1: cat.
            # En yüksek skora sahip olan sınıfı (0: Uçak veya 1: Kedi) seç
            _, tahminler = torch.max(outputs, 1) #torchmax aslinda iki deger dondurur: en yuksek skor ve onun indexi. vize sadece indexi lazim.
            
            # o anki banchte kac resim varsa sayaca ekle. labels.size(0) ile batchteki resim sayisini aliyoruz. labels tensor’ünün 0. boyutunun uzunluğunu verir.
            toplam_resim += labels.size(0)
            dogru_bilinen += (tahminler == labels).sum().item() # tahminler ve labels karsilastirilir. dogru tahminler True, yanlis tahminler False olur. sum() ile True sayisi bulunur. item() ile tensor'den sayiya cevrilir.
            #Her 10 pakette bir anlık durumu yazdır
            if (i + 1) % 10 == 0:
                print(f"Test Ediliyor... Paket [{i+1}/{len(test_loader)}]")
    # 5. Matematiksel Doğruluk Yüzdesi Hesaplama 
    basari_orani = 100.0 * dogru_bilinen / toplam_resim
    print(f"Test Bitti! Modelin Basari Orani: %{basari_orani:.2f}")

if __name__ == "__main__":
    test_model()