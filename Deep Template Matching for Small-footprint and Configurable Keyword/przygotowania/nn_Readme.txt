--brak dobrego opisu warstwy drugiej

Schemat architektury sieci neuronowej; przestawia w sposób zrozumiały przepływ danych w sieci.

Założenia:
T - liczba próbek w badanym pliku

wejściem dla systemu jest 128-band spektrogram w skali mel (słyszalności) czyli macierz 2 wymiarowa 128xT

System słada się z 3 części:

1. Feature extractor - konwolucyjna sieć neuronowa dwuwymiarowa (conv2d), która posiada 6 warstw nie licząc przekształceń macierzy. Pierwsze 3 warstwy to konwolucyjne dwuwymiarowe, następnie warstwa max-pool, liniowa (gęsta) oraz rekurencyjna jednostka gru (dokładny opis rozmiarów ->tabela 1 strona 2 Deep Template Matching for Small-footprint and Configurable Keyword Spotting.pdf)

2. Template matching module - czerpie z wyjść z ekstraktora dla badanego pliku oraz dla przykłądowego wypowiedzenia frazy kluczowej.
Na początku obliczane jest seq2seq attention dla tych wyjść modułu porzedniego, następnie obliczny jest dystans pomiędzy wzorcem i znalezionym podobieństwem. Na końcu dopasowywane są te odległości w 
co ja piszę

3. Binary classifier - składa się z z sieci gęstej z jedną warstwą ukrytą dającą wyjście w postawci 2 wymiarowego wektora pozwalącego stwierdzić wystąpienie lub jego brak w badanym fragmencie mowy.
(Punkt 2.2 strona 2 Deep Template Matching for Small-footprint and Configurable Keyword Spotting.pdf)
nn.png (wc Wykonano za pomocą narzędzia NN-SVG autorstwa Alex Lenail
https://github.com/alexlenail/NN-SVG