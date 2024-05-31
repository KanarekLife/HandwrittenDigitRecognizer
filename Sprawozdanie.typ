#set text(
  font: "Calibri Light",
  size: 11pt
)
#set page(paper: "a4", margin: (x: 1cm, y: 1cm), numbering: "1")
#set heading(numbering: "1.")
#set figure(supplement: none)
#set grid(columns: 2, gutter: 2mm,)
#set pad(x: -0.8cm)

#align(horizon)[
  #align(center)[
    #stack(
      v(12pt),
      text(size: 20pt)[Sztuczna inteligencja],
      v(10pt),
      text(size: 18pt)[MNIST i analiza metod klasyfikacji cyfr],
      v(15pt),
      text(size: 14pt)[Jakub Bronowski, 193208],
      v(10pt),
      text(size: 14pt)[Bartłomiej Krawisz, 193319],
      v(10pt),
      text(size: 14pt)[Stanisław Nieradko, 193044],
      v(15pt),
      text(size: 12pt)[gr. 2, sem. IV]
    )
  ]
]

#pagebreak()

#align(top)[
#outline(title: "Spis treści", target: heading.where(depth: 1))
]
#pagebreak()

= Opis zadania
Jako projekt zaliczeniowy z przedmiotu Sztuczna inteligencja zdecydowaliśmy się na analizę różnych metod rozpoznawania cyfr pisanych odręcznie. Celem projektu jest ich porównanie. W projekcie użyto 5 metod: k-najbliższych sąsiadów (ang. k-nearest neighbors), liniowa maszyna wektorów nośnych (ang. linear SVM recogniser), nieliniowa maszyna wektorów nośnych (ang. non-linear SVM recogniser), losowy las decyzyjny (ang. random decision forests) oraz sieć neuronowa (ang. neural network). Projekt zrealizowano w języku Python, wykorzystując biblioteki: numpy, matplotlib, scikit-learn, tensorflow, keras. 
= Opis zbioru danych
Zbiór danych to MNIST opracowany przez National Institute of Standards and Technology (agencja rządowa USA odpowiedzialna za rozwój i promocję produktów przemysłu USA). Wykorzystano wersję dostępną w bibliotece `pytorch`. Zbiór ten zawiera łącznie 70 000 obrazów przedstawiających cyfry o rozmiarze 28x28 pikseli pisanych odręcznie, białym tuszem na czarnym tle. 60 000 z nich to dane treningowe (pobierane z pliku `train-images-idx3-ubyte`) a pozostałe 10 000 to dane testowe (pobierane z pliku `t10k-images-idx3-ubyte`). Dane z obrazu są interpretowane jako macierz 26 x 26 z wartościami pikseli w skali szarości o kolorze z zakresu [0, 255], gdzie 0 oznacza kolor czarny, a 255 kolor biały. Każda cyfra jest przypisana do jednej z 10 klas oznaczających jej wartość [0, 9].


= Opis badanych metod
== K-najbliżsi sąsiedzi
Metoda k-najbliższych sąsiadów (ang. k-nearest neighbors, KNN) jest jedną z najprostszych metod klasyfikacji. 
Podjęcie decyzji o przynależności do klasy oparte jest na ocenie przynależności do klas k najbliższych punktów ze zbioru referencyjnego w przestrzeni cech. Użyto metody `KNeighborsClassifier()` z biblioteki `scikit-learn`. Wybrano domyślne parametry dla trenowania modelu, w szczególności parametr `n_neighbors` ustawiono na 5. Według literatury, metoda dobrze sprawdza się w przypadku problemów klasyfikacji, gdzie granica decyzyjna jest złożona i nieregularna. 

== Liniowa maszyna wektorów nośnych
Liniowa maszyna wektorów nośnych (Linear Support Vector Machine) to popularny algorytm klasyfikacji, który stara się znaleźć najlepszą linię, która maksymalnie oddziela klasy danych. Algorytm stara się znaleźć taką linię, która rozdziela dane na różne klasy maksymalizując marginesy, czyli odległości między linią a najbliższymi punktami danych z każdej klasy. Użyto metody `SVC()` z biblioteki `scikit-learn` z parametrem `kernel='linear'`. Uczenie polega na maksymalizacji marginesu przy jednoczesnym minimalizowaniu wartości kar dla źle rozdzielonych klas.

== Nieliniowa maszyna wektorów nośnych
Nieliniowa maszyna wektorów nośnych (Non-linear Support Vector Machine) to rozszerzenie liniowej maszyny wektorów nośnych, które pozwala na rozdzielenie danych nieliniowych. W tym celu wykorzystuje się funkcję jądra (ang. kernel), która mapuje dane do przestrzeni o wyższej wymiarowości, licząc na to, że dane są w niej liniowo separowalne. Użyto metody `SVC()` z biblioteki `scikit-learn` z parametrem `kernel='rbf'` - przyjęto jądro Gaussa, `C=10` - skala regularyzacji, `gamma=0.001` - współczynnik jądra. Uczenie polega na znalezieniu hiperpłaszczyzny, która najlepiej separuje dane w przestrzeni cech. Według literatury, metoda ta dobrze sprawdza się w przypadku danych nieliniowo separowalnych o skomplikowanej strukturze.

== Losowy las decyzyjny
Losowy las decyzyjny (Random Decision Forests) to metoda klasyfikacji, która polega na zbudowaniu wielu drzew decyzyjnych i wybraniu klasy, która jest najczęściej wybierana przez poszczególne drzewa (zasada "mądrości tłumu", każda próbka do oceny jest analizowana przez każde z drzew). Drzewa są trenowane na podzbiorze losowo wybranych cech przy jednoczesnej redukcji overfittingu. Użyto metody `RandomForestClassifier()` z biblioteki `scikit-learn` z domyślnymi parametrami. Według literatury, metoda ta dobrze sprawdza się w przypadku dużych zbiorów danych, gdzie granica decyzyjna jest złożona i nieregularna.

== Sieć neuronowa
Sieć neuronowa (Neural Network) to model inspirowany biologicznymi neuronami, pogrupowanymi w wiele warstw. Sieć jest trenowana na danych uczących, a proces ten polega na dostosowaniu wag między neuronami w taki sposób, aby zminimalizować błąd predykcji. Według literatury, sieci neuronowe dobrze sprawdzają się przy rozpoznawaniu obiektów i rzekomo gwarantują najlepszą trafność. Sieć neuronowa powstała przy użyciu metod z biblioteki `pytorch`. Posiada ona 2 warstwy konwolucyjne. Pamatery pierwszej warstwy to: `input_channels=1`, `output_channels=32`, `kernel_size=3x3`. Parametry drugiej warstwy to: `input_channels=32`, `output_channels=64`, `kernel_size=3`. Następne 2 wartswy odrzucają odpowiednio 25% i 50% danych. Następne 2 warstwy są liniowe (liniowe transformacje danych z wykorzystaniem uprzednio wyznaczonych wag i biasów), z odpowiednio 9212 i 128 wejściami oraz 128 i 10 wyjściami. Wybraną funkcją aktywacji jest `Relu`. Wykorzystano także optymalizator `Adadelta` z domyślnymi parametrami. 

= Trening modeli !(wyliczanie score() - na jakich danych)
Kazdy z modeli został wytrenowany na tym samym zbiorze danych treningowych. Zastosowano ustawienia metod identyczne, jak opsiane powyżej. Wykorzystano dane treningowe z MNIST. Dla wszystkich badanych metod zbiór danych teningowych to 60 000 obrazów. Trenowanie odbywa się przy pomocy metody `fit()`. Ocenę treningu przeprowadza się poprzez wywołanie metody `score()` na danych treningowych z zestawy MNIST, który wypisuje się w konsoli. Tworzy się także raport z trenowania `training_report.txt`. 
Sieć neuronowa została wytestowana na 14 epokach. Dane treningiwe takie same jak dla pozostałych metod.

== Raport z treningu 

= Testowanie modeli
Dla każdej z metod po uprzednim wytrenowaniu wyznacza się macierz pomyłek, reprezentowaną graficznie. Im bardzej fioletowe punkty poza główną przekątną, tym większa liczba błędów. Na przecięciu wiersza i kolumny widać liczbę obrazów, które zostały zaklasyfikowane jako cyfra z wiersza, a były w rzeczywistości cyfrą z kolumny. Macierz pomyłek powstałą na podstawie dancyh testowych z MNIST (10 000 obrazów). Istnieje też możliowość wytestowania modelu na własnych danych, w folderze `test_data` należy umieścić obrazy do sprawdzenia. Wymaga się, aby cyfry były namalowane czarnym kolorem na białym tle, a obrazek był w proporcjach 1:1 (kwadrat). Wyniki testowania zapisywane są w pliku `testing_report.txt`.

== Macierz pomyłek
== Raport z testowania ręcznego

== Opis implementacji metody bazowej
== Opis poczynionych zmian / Prób zmian: parametry
== Wyniki (accuracy, confusion matrix, czas trenowania, czas predykcji)

= Implementacja aplikacji
== Opis aplikacji
== Metody optymalizacji / normalizacji obrazu
== Podsumowanie i wyniki w praktyce

= Podsumowanie




== Wzorki metematyczne
$ phi.alt_i (x) = limits(product)_(j=1,j!=i)^(n+1) frac((x - x_j), (x_i - x_j)) $
$ F(x) = sum_(i=1)^(n+1) y_i phi.alt_i (x) $
$ x_k = frac(a + b, 2) + frac(b - a, 2) cos(frac(2k - 1, 2k) pi), k = 1, ..., n $