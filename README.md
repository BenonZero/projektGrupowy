# projektGrupowy
 
## Spis treści
* [Informacje ogólne](#informacje-ogólne)
* [Technologie](#technologie)
* [Setup](#setup)
* [Użycie](#użycie)

## Informacje ogólne
Ten projekt skład się z opracowań trzech metod wykrywania słów kluczowych w mowie ciągłej zamieszonych w odpowiadających folderach oraz aplikacji .z przygotowanymi modelami gotowej do uruchomienia w folderze Aplikacje.
	
## Technologie
Projekt tworzony przy pomocy:
* Python wersja: 3.10
* torch library version: 2.0.1
	
## Setup
Aby uruchomić aplikacje należy zanstalować program Python oraz bibliotekę pytorch a następnie uruchomić wybraną aplikację z folderu Aplikacje.
Aby rozpocząć uczenie należy pobrać pliki main oraz model z wybranego rodzaju modelu oraz uruchomić plik main; aby wybrać własne dane do uczenia konieczne jest zmodyfikowanie  zmiennych test_set, oraz test_set aby zawierały wybrany zbiór danych. 

## Użycie
1. Najpierw należy wybrać mdoel
2. Po wybraniu pierwszego modelu należy wybrać słowo szukane
3. W przypadku wybrania drugiego modelu nie trzeba wybierać słowa
4. Należy wybrać czy fraza ma być szukana w pliku dźwiękowym czy z mikrofonu
5. Po każdorazowej próbie lub niepoprawnym wyborze należy uruchomić ponownie aplikację
