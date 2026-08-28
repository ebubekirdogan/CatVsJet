# train_final.py -> Optuna'nin bulundugu en iyi hiperparametrelerle Model 3'u egiten dosya.
# optuna_search.py TAMAMLANDIKTAN SONRA, sadece bir kere calistirilir.
#
# epochs=5 verdik cunku Model 1 ve Model 2 de 5 epoch egitildi - adil karsilastirma 
# ucunun de ayni epoch sayisinda olmasi lazim. 
# save_files=True ve verbose=True verdik
# cunku bu artik gercek, kaydedilecek olan final model.

from train import train_model

if __name__ == "__main__":
    # asagidaki iki degeri optuna_search.py'nin ciktisindaki "En iyi parametreler" ile degistiriyoruz
    best_learning_rate = 0.00037555364094979084
    best_batch_size = 16

    print(f"Final Model 3 egitimi basliyor (Optuna'nin buldugu en iyi degerlerle)\n")

    history, val_acc = train_model(
        learning_rate=best_learning_rate,
        batch_size=best_batch_size,
        epochs=5,
        save_files=True,
        verbose=True
    )

    print(f"\nFinal model egitimi bitti. Son epoch val accuracy: {val_acc:.2f}%")