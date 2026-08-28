# optuna_search.py -> En iyi learning_rate ve batch_size kombinasyonunu Optuna ile arayan dosya.
# train_model fonksiyonunu import ediyoruz train.py dosyasından, Optuna her denemede bu fonksiyonu cagiracak.

import optuna
from train import train_model

# objective fonksiyonu -> Optuna her denemede (trial) bu fonksiyonu cagirir.
# trial nesnesi, Optuna'nin "bu denemede su degerleri sec" dedigi yer.
def objective(trial):
    # learning_rate: 0.00001 ile 0.1 arasinda, log olcekte ariyoruz (log=True).
    # log=True cunku 0.001 ile 0.01 arasi da 0.01 ile 0.1 arasi kadar onemli olmali,
    # normal (linear) aramada buyuk sayilara haksiz yere daha cok agirlik verilirdi.
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    # batch_size: sadece bu 3 degerden birini secebilir (rastgele herhangi bir sayi degil,
    # GPU bellek sinirlari yuzunden genelde 2'nin katlari kullanilir).
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # bu denemeyi KISA tutuyoruz (epochs=2) - amac en iyi bolgeyi hizlica bulmak,
    # save_files=False, verbose=False -> 15 denemenin hicbirini diske yazmiyoruz, ekrani doldurmuyoruz.
    history, val_acc = train_model(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=2,
        save_files=False,
        verbose=False
    )

    # Optuna bu sayiya bakarak "bu deneme iyiydi mi kotu muydu" diye anlayacak.
    return val_acc

if __name__ == "__main__":
    # direction='maximize' -> val_acc'i BUYUTMEYE calisiyoruz (kucultmeye degil, o yuzden onemli)
    study = optuna.create_study(direction='maximize')

    # objective fonksiyonunu n_trials kadar (15 kere) cagirir, her seferinde farkli deger dener.
    print("Optuna aramasi basliyor (15 deneme, her biri 2 epoch)...\n")
    study.optimize(objective, n_trials=15)

    print("\n========== OPTUNA SONUCLARI ==========")
    print(f"En iyi val accuracy : {study.best_value:.2f}%")
    print(f"En iyi parametreler : {study.best_params}")
    print("=======================================")

    # en iyi parametreleri bir txt dosyasina da yaziyoruz, train_final.py'de buradan okuyup kullanacagiz
    with open('results/optuna_best_params.txt', 'w', encoding='utf-8') as f:
        f.write(f"En iyi val accuracy: {study.best_value:.2f}%\n")
        f.write(f"En iyi parametreler: {study.best_params}\n")

    print("\nSonuclar 'results/optuna_best_params.txt' dosyasina kaydedildi.")