import os
from ultralytics import YOLO
from scripts.gestures import Gesture10

model3 = YOLO('runs/classify/train3epochs/weights/best.pt')
model10 = YOLO('runs/classify/train10epochs/weights/best.pt')
model20 = YOLO('runs/classify/train20epochs/weights/best.pt')

sciezka_do_zdjecia = '../gesty/Wizja gesty/Maciej/tlo_1'

modele_do_testu = {
    "Model (3 epoki)": model3,
    "Model (10 epok)": model10,
    "Model (20 epok)": model20
}

podsumowanie_modeli = {}

for nazwa_modelu, model in modele_do_testu.items():
    results = model.predict(sciezka_do_zdjecia, verbose=False)

    poprawne_ogolem = 0
    wszystkie_ogolem = len(results)
    suma_pewnosci_ogolem = 0.0

    statystyki_klas = {gest.name: {'poprawne': 0, 'wszystkie': 0, 'suma_pewnosci': 0.0} for gest in Gesture10}

    print(f"\n==================================================")
    print(f" WYNIKI DETEKCJI: {nazwa_modelu.upper()}")
    print(f"==================================================")

    for result in results:
        sciezka_pliku = result.path
        nazwa_pliku = os.path.basename(sciezka_pliku)

        litera = nazwa_pliku[0].lower()
        numer_klasy = ord(litera) - ord('a') + 1
        oczekiwany_gest = Gesture10(numer_klasy).name

        top_class_id = result.probs.top1
        yolo_wynik = result.names[top_class_id]
        rozpoznany_gest = Gesture10(int(yolo_wynik)).name
        pewnosc = result.probs.top1conf.item() * 100

        statystyki_klas[oczekiwany_gest]['wszystkie'] += 1
        statystyki_klas[oczekiwany_gest]['suma_pewnosci'] += pewnosc
        suma_pewnosci_ogolem += pewnosc

        if oczekiwany_gest == rozpoznany_gest:
            poprawne_ogolem += 1
            statystyki_klas[oczekiwany_gest]['poprawne'] += 1
            status = "✅"
        else:
            status = "❌"

        print(
            f"{status} Plik: {nazwa_pliku:15} | Oczekiwany: {oczekiwany_gest:15} | Model: {rozpoznany_gest:15} | Pewność: {pewnosc:.2f}%")

    print("\n--- SKUTECZNOŚĆ NA KLASĘ ---")
    for nazwa_klasy, statystyki in statystyki_klas.items():
        if statystyki['wszystkie'] > 0:
            skutecznosc = (statystyki['poprawne'] / statystyki['wszystkie']) * 100
            srednia_pewnosc = statystyki['suma_pewnosci'] / statystyki['wszystkie']
            print(
                f"{nazwa_klasy:15} | Skuteczność: {skutecznosc:6.2f}% | Śr. pewność: {srednia_pewnosc:6.2f}% | Trafienia: {statystyki['poprawne']:2}/{statystyki['wszystkie']:2}")
        else:
            print(f"{nazwa_klasy:15} |   ---  | Brak zdjęć testowych")

    skutecznosc_ogolna = (poprawne_ogolem / wszystkie_ogolem) * 100 if wszystkie_ogolem > 0 else 0
    srednia_pewnosc_ogolna = suma_pewnosci_ogolem / wszystkie_ogolem if wszystkie_ogolem > 0 else 0

    podsumowanie_modeli[nazwa_modelu] = {
        'skutecznosc': skutecznosc_ogolna,
        'srednia_pewnosc': srednia_pewnosc_ogolna
    }

    print(f"\n--- PODSUMOWANIE OGÓLNE ---")
    print(f"Poprawnie rozpoznano:  {poprawne_ogolem} z {wszystkie_ogolem} zdjęć")
    print(f"Całkowita skuteczność: {skutecznosc_ogolna:.2f}%")
    print(f"Średnia pewność:       {srednia_pewnosc_ogolna:.2f}%\n")

print(f"\n==================================================")
print(f" OSTATECZNE PORÓWNANIE MODELI")
print(f"==================================================")
for nazwa, stat in podsumowanie_modeli.items():
    print(f"{nazwa:17} | Skuteczność: {stat['skutecznosc']:6.2f}% | Śr. pewność: {stat['srednia_pewnosc']:6.2f}%")