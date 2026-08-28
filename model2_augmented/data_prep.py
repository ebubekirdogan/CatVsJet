### data_prep.py -> Resimleri modelin anlayabilecegi bicime getirmek
## ===================
#  (IMPORTS)
# pytorch ana motoru
import torch
# datasets; klasor icerisindeki resimleri okuma
# transforms; resimlerin boyutunu degistirme, tensor'e cevirme gibi islemler icin
from torchvision import datasets, transforms
# Dataloader verileri dataloader batch'ler halinde 32serli okuma işlemi için
from torch.utils.data import DataLoader
# isletim sistemi ile ilgili islemler icin
import os
#  ===================
# artik val icin ayri, hazir bir klasor var (train/val/test), o yuzden train'i kod icinde
# rastgele bolmeye (random_split, val_split, seed) gerek kalmadi, dogrudan 3 klasoru okuyoruz.
def get_dataloaders(data_dir='../dataset', batch_size=32):
    # veri on isleme
    # ============================================
    # veri artirma (augmentation) - SADECE train icin. model her epoch'ta ayni resmi
    # hafifce degistirilmis haliyle gorsun diye, ezberlemesi zorlassin diye ekliyoruz.
    # ============================================
    train_transform = transforms.Compose([
        transforms.Resize((64,64)), # farklı boyurrutlarda resimleri 64x64 boyutuna getiriyoruz. Tum resimler artik 64x64x3(3 RGB den)
        transforms.RandomHorizontalFlip(), # resmi %50 ihtimalle yatay aynalar (sag-sol ters cevirir)
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # parlaklik ve kontrasti rastgele +-%30 oynatir (amirin bahsettigi parlaklik artirma)
        transforms.RandomRotation(15), # resmi rastgele +-15 derece dondurur
        transforms.ToTensor(), # resimleri tensor'e çeviriyoruz
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # resimleri normalize ediyoruz Amac : Resim değerlerini modelin daha rahat işleyebileceği aralığa getirirmektir.
    ])

    # val ve test icin augmentation YOK, cunku bunlar modelin GERCEK dunyada nasil performans
    # gosterdigini olcmeli, yapay olarak degistirilmis resimlerle olculmemeli.
    val_test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# klasor yollari

    #isletim sistemine uygun sekilde klasor yollari olusturuyoruz
    train_dir = os.path.join(data_dir, 'train') #Windows;Mac ve Linux ayrimi icin os.path.join kullaniyoruz.(dosya yollarindaki / ve \ farki icin)
    val_dir = os.path.join(data_dir, 'val') # validation klasoru artik ayri, hazir bir klasor
    test_dir = os.path.join(data_dir, 'test')

    # Resimleri dataset olarak okuma
    # ImageFolder; verilen klasor icindeki her alt klasoru otomatik olarak sinif olarak algilar. klasor ismine goer etiket verir
    # train_dataset augmentation'li transform ile, val/test augmentation'siz transform ile okunuyor
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    # data loaer - resimleri batchler halinde (32 li gruplar halinde) modele verme islemi gereceklestirilir.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 3 boyutlu resimlerden 32 tanesini ust uste koyarak 4 boytlu matris hazirlar.# model ezber yapmasin diye shuffle=True yaptik.resimleri karisik verir
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # validation'i karistirmiyoruz, sadece olcum yapiyoruz, ogrenme yok
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # test verilerini karistirmiyoruz. ogrenme islemi yapilmayacak

        # ciktilar

    print(f"Egitim: {len(train_dataset)} resim | Validation: {len(val_dataset)} resim | Test: {len(test_dataset)} resim hazir")

    return train_loader, val_loader, test_loader # model egitim dosyasında cagirilacak.

if __name__ == "__main__":
    get_dataloaders()

# if __name__ == "__main__":
#Her .py dosyasının gizli bir __name__ değişkeni vardır.
#Dosya doğrudan çalıştırılırsa (python data_prep.py) → __name__ = "__main__" olur.
#Dosya başka bir dosyadan import edilirse (from data_prep import ...) → __name__ = "data_prep" (dosyanın kendi adı) olur.
#if __name__ == "__main__": bloğu, sadece dosya doğrudan çalıştırıldığında çalışır; import edildiğinde çalışmaz.
#Amaç: import edildiğinde dosyanın alt satırlarındaki fonksiyon çağrısının (get_dataloaders()) istemsizce tekrar çalışmasını engellemek → gereksiz tekrar işlem / gereksiz print çıktısı olmasın diye.