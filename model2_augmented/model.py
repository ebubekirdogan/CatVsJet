# bu dosyada modelin mimarisi tanimlanacak. modelin katmanlari ve ileri yayilim islemi burada olacak.
#=================
# CNN Architecture
#=================
import torch
import torch.nn as nn #katmanların bulundugu kutuphane (conv2d, maxpool2d gibi)
import torch.nn.functional as F # fonksiyonların bulundugu kutuphane (ReLU burada)

class CatVsJetCNN(nn.Module): #pytorch'da model olusturmak icin nn.Module(sinir agi modelleri temel sinifi) sinifindan miras aliyoruz
    def __init__(self):
        super(CatVsJetCNN, self).__init__() #Üst sınıf olan nn.Module sınıfının başlangıç işlemlerini çalıştırır.
        

        # 1. evrisim katmani
        #Giris resmi RGB oldugundan 3 kanalli. resimden 16 tane filtre ile evrisim yapacagiz. kernel boyutu 3x3 olacak. padding=1 ile resim boyutunu koruyoruz.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        # ilk katman daha basit seyler yakalar 16 filtre yeterli. 2. katman daha karmasık seyler yakalar 32 filtre olsun.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 

        # Pooling katmani; maxpooling ile her bloktaki en büyük deger alinip digerleri atilir. kernel_size ile 2x2 lik pencerelerle bakilir. stride=2 ile pencereler 2 birim kaydirilir. boyutu yarıya indirir.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # max pooling ile boyutu yarıya indiriyoruz

        # Fully connected katmani;fc1 8192 sayiyi alip 512 sayiya indiriyor. fc2 512 sayiyi alıp 2 sayiya indiriyor. 2 sayi; airplane ve cat siniflari icin.
        # Hidden layer       
        self.fc1 = nn.Linear(in_features=8192, out_features=512) #8192 den 2 ye agresif bir sikistirma yerinde once 512 ye sikistiriyoruz.
        #*** 8192 cunku 2 conv 2 pooling islemi yapilinca giris 3x64x64 den 32x16x16 ya geliyor (en bastaki kanal conv de artan kisim. ve pooling de yariya inen kisim ise matrisin boyutu. 32x32 den 16x16 ya innesi gibi.)***#
        # Output layer
        self.fc2 = nn.Linear(in_features=512, out_features=2) 

    def forward(self, x): # x ; modele giren veri.bir batch'lik resim tensor'ü, örn. [32, 3, 64, 64] boyutunda — 32 resim, 3 kanal, 64x64). 
        x = self.conv1(x) # resmi 1. evrisim katmanina veriyoruz. 3 kanalli resimden 16 kanalli resim cikiyor. boyut [32, 16, 64, 64] oluyor.Sonuc tekrardan x e ataniyor.
        x = F.relu(x)     # Reludan gecer  
        x = self.pool(x)  # max pooling ile boyutu yarıya indiriyor. boyut [32, 16, 32, 32] oluyor.

        x = self.conv2(x) # 2. evrisim katmanina veriyoruz. 16 kanalli resimden 32 kanalli resim cikiyor. boyut [32, 32, 32, 32] oluyor.
        x = F.relu(x)     
        x = self.pool(x)  # max pooling ile boyutu yarıya indiriyor. boyut [32, 32, 16, 16] oluyor.

        x = x.view(-1, 8192) # FLATTEN adimi: 4 boyutlu tensorden duzlestirip tek sira haline getirir.

        x = self.fc1(x)   # 8192 sayiyi aliyor ve 512 sayiya indiriyor. boyut [32, 512] oluyor.
        x = F.relu(x)     # Reludan gecer

        x = self.fc2(x)   # 512 sayiyi aliyor ve 2 sayiya indiriyor. boyut [32, 2] oluyor. 2 sayi; airplane ve cat siniflari icin.
        return x          

#==========Katman ile Fonksiyon farkı===================
# Katmanların (conv2d,linear gibi)içinde öğrenilebilir ağırlıklar (weight) var, bu yüzden self. ile __init__ içinde tanımlanıp saklanmaları gerekiyor.
# ReLU gibi işlemlerin öğrenilecek hiçbir ağırlığı yok, sadece "gelen sayıyı şu kurala göre dönüştür" diyor — o yüzden F.relu(x) şeklinde direkt forward içinde çağırman yeterli, ayrıca saklamana gerek yok.
