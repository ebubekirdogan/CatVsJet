### data_prep.py -> Resimleri modelin anlayabilecegi bicime getirmek
## ===================
#  (IMPORTS)
# pytorch ana motoru
import torch
# datasets; klasor icerisindeki resimleri okuma
# transforms; resimlerin boyutunu degistirme, tensor'e cevirme gibi islemler icin
from torchvision import datasets, transforms
# Dataloader verileri dataloader batch'ler halinde 32serli okuma işlemi için
from torch.utils.data import DataLoader, random_split
# isletim sistemi ile ilgili islemler icin
import os
#  ===================
# val_split: train verisinin yuzde 20 si ayrilacagi icin validationa 0.2, seed ise her calistirmada ayni bolunmeyi elde etmek icin 42 secildi. (her calistirmada farkli resimler train/val'a duserdi, bu da sonuclari karsilastirilamaz yapardi.42 nin ozel bir anlami yok,sbt sayi)
def get_dataloaders(data_dir='./dataset', batch_size=32, val_split=0.2, seed=42):
    # veri on isleme
    transform = transforms.Compose([
        transforms.Resize((64,64)), # farklı boyurrutlarda resimleri 64x64 boyutuna getiriyoruz. Tum resimler artik 64x64x3(3 RGB den)
         transforms.ToTensor(), # resimleri tensor'e çeviriyoruz
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # resimleri normalize ediyoruz Amac : Resim değerlerini modelin daha rahat işleyebileceği aralığa getirirmektir.
])

# klasor yollari

    #isletim sistemine uygun sekilde klasor yollari olusturuyoruz
    train_dir = os.path.join(data_dir, 'train') #Windows;Mac ve Linux ayrimi icin os.path.join kullaniyoruz.(dosya yollarindaki / ve \ farki icin)
    test_dir = os.path.join(data_dir, 'test')

    # Resimleri dataset olarak okuma
    # ImageFolder; verilen klasor icindeki her alt klasoru otomatik olarak sinif olarak algilar. klasor ismine goer etiket verir
    # Yukarida tanimladigimiz transform islemlerini resimlere uygular
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # ============================================
    # VALIDATION SPLIT
    # ============================================
    # full_train_dataset icinde kac resim oldugunu buluyoruz
    val_size = int(len(full_train_dataset) * val_split) # traindeki %20 resim validation'a ayrilacak
    train_size = len(full_train_dataset) - val_size # geri kalan resimler gercek train seti olacak

    # random_split; verdigimiz dataset'i verdigimiz boyutlarda rastgele parcalara ayirir.
    # generator=torch.Generator().manual_seed(seed) -> her kod calistiginda AYNI rastgele bolunmeyi elde etmemizi saglar.
    # (seed olmasaydi her calistirmada farkli resimler train/val'a duserdi, bu da sonuclari karsilastirilamaz yapardi.)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    # ============================================

    # data loaer - resimleri batchler halinde (32 li gruplar halinde) modele verme islemi gereceklestirilir.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 3 boyutlu resimlerden 32 tanesini ust uste koyarak 4 boytlu matris hazirlar.# model ezber yapmasin diye shuffle=True yaptik.resimleri karisik verir
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # validation'i karistirmiyoruz, sadece olcum yapiyoruz, ogrenme yok
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # test verilerini karistirmiyoruz. ogrenme islemi yapilmayacak

        # ciktilar

    print(f"Sınıflar(Etiketler): {full_train_dataset.classes}") # ImageFolder sinifinin otomatik olarak cikarttigi sinif listesi.
    print(f"Egitim: {train_size} resim | Validation: {val_size} resim | Test: {len(test_dataset)} resim hazir")

    return train_loader, val_loader, test_loader # model egitim dosyasında cagirilacak.

if __name__ == "__main__":
    get_dataloaders()

# if __name__ == "__main__":
#Her .py dosyasının gizli bir __name__ değişkeni vardır.
#Dosya doğrudan çalıştırılırsa (python data_prep.py) → __name__ = "__main__" olur.
#Dosya başka bir dosyadan import edilirse (from data_prep import ...) → __name__ = "data_prep" (dosyanın kendi adı) olur.
#if __name__ == "__main__": bloğu, sadece dosya doğrudan çalıştırıldığında çalışır; import edildiğinde çalışmaz.
#Amaç: import edildiğinde dosyanın alt satırlarındaki fonksiyon çağrısının (get_dataloaders()) istemsizce tekrar çalışmasını engellemek → gereksiz tekrar işlem / gereksiz print çıktısı olmasın diye.