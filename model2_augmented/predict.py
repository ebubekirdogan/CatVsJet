# predict.py -> Tek bir resmi modele gosterip tahmin aldigimiz demo dosyasi.
# Bu dosyanin, egitim/test sureciyle ilgisi olmayip, sadece "modelimiz gercekte calisiyor mu"
# diye gostermek icin yazilmistir.

import torch #PyTorch kutuphanesi
from PIL import Image # resim dosyalarini acmak icin Pillow kutuphanesi
from torchvision import transforms #transform islemleri icin.

from model import CatVsJetCNN

# ============================================
# test edilecek resmin dosya yolu
# ============================================
IMAGE_PATH = "image4.jpg" 

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bos model olustur, egitilmis agirliklari yukle 
    model = CatVsJetCNN().to(device)
    model.load_state_dict(torch.load('cat_vs_jet_model2_augmented.pth', weights_only=True))
    model.eval()

    # data_prep.py'deki val/test transform ile AYNI olmali (augmentation YOK, cunku bu
    # gercek bir tahmin, egitim degil - resmi oldugu gibi degerlendirmek istiyoruz)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # PIL ile resmi ac, RGB'ye cevir.
    image = Image.open(image_path).convert('RGB')

    # transform uygula, sonra modele tek bir resim olarak degil "1 elemanli batch" olarak vermemiz lazim
    # unsqueeze(0) -> [3, 64, 64] boyutundaki tensore basa 1 boyut ekler, [1, 3, 64, 64] yapar
    # (model, DataLoader'dan hep batch halinde resim gormeye alisik, tek resim versek bile bu sekle sokmamiz gerekiyor)
    image_tensor = transform(image).unsqueeze(0).to(device)

    class_names = ['airplane', 'cat']  # ImageFolder'in alfabetik sirada verdigi sinif isimleri

    with torch.no_grad():  # tahmin yapiyoruz, ogrenme yok
        outputs = model(image_tensor)  # ham skorlar, orn: [1.2, 4.7]

        # softmax, ham skorlari (-sonsuz, +sonsuz) araligindan 0-1 arasi "olasilik" gibi
        # yorumlanabilir sayilara cevirir, toplamlari 1.
        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)  # en yuksek skorlu sinifi sec
        confidence = probabilities[0][predicted.item()].item()  # probabilities ile sinifin olasiligini al

    predicted_class = class_names[predicted.item()]

    print(f"Resim: {image_path}")
    print(f"Tahmin: {predicted_class}")
    print(f"Guven (confidence): %{confidence*100:.2f}")
    print(f"Tum olasiliklar -> airplane: %{probabilities[0][0].item()*100:.2f} | cat: %{probabilities[0][1].item()*100:.2f}")

if __name__ == "__main__":
    predict_image(IMAGE_PATH)