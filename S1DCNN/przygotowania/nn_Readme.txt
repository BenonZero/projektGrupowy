Schemat architektury sieci neuronowej; przestawia w sposób zrozumiały przepływ danych w sieci.

Założenia:
N(liczba filtrów)=32
T= liczba próbek MFCC
K(rozmiar filtra) = 9
L(zakres przyszłych obserwacji) = 2




Wejściem dla warstwy pierwszej jest 13 wektorów MFCC o długości T, gdzie T to liczba próbek w badanym pliku dźwiękowym.

Wyściem warstwy pierwszej są 32 wektory(dla rozpatrywania przetwarzania w czasie) dla 32 filtrów warstwy konwolucyjnej. Każdy filtr dla wyznaczenia wyniku w czasie t1 otrzymuje wejście wartości wszystkich wielkości MFCC w czasie(lub pozycji w wektorze) t1. 

Wejściem dla warstwy drugiej są wyjścia warswty pierwszej takie, że każdy z filtrów warstwy drugiej korzysta z tylko z wyjść odpowiadającego wektora w warstwie pierwszej (np. filtr numer 4 w warstwie drugiej korzysta tylko z wyjść filtra numer 4 w warstwie pierwszej). Wykorzystywane jest K-L-1 poprzednich wyjść, aktualne oraz L następnych wyjść.
--Wykorzystywane w analizowanym przykładie wartości to K(rozmiar filtra) równe 9, L(zakres przyszłych obserwacji) równe 2.

Wyjściem warstwy drugiej są 32 wektory o długości T

Należy zauważyć że warsty pierwsza i druga są złożone na sobie w liczbie siedmiu!

Wszystkie te wyjścia warstw drugich trafiają do pojedynczej warstwy gęstej która daje nam wynik w postaci (wedle przykładu) 20 logitów dla 20 badanych klas(słów) a na końcu za pomocą funcji SoftMax otrzymujemy procentowe oszacowanie prawdopodobieństwa wystąpienia danej klasy.


nn.png (wcześniej .svg) Wykonano za pomocą narzędzia NN-SVG autorstwa Alex Lenail
https://github.com/alexlenail/NN-SVG
