import modelik
import input


number_of_vectors_layer_1 = 13                  #wejście warstwy 1 - 13 wektorów długości T
number_of_filters = 32                          #liczba filtrów w każdej warstwie
number_of_vectors_layer_2 = 32                  #1 wektor dla każdego filtra ^  (filtr 4 w warstwie 2 kożysta tylko z wyjścia filtra 4 w warstwie pierwszej)
number_of_logits=20                             #liczba logitów na wyjściu
kernel_size = 9
number_of_networks = 7


mfccs_tensor = input.crete_mfccs_vectors()
print(mfccs_tensor)
model = modelik.Stacked1DCNN(number_of_vectors_layer_1, number_of_vectors_layer_2, number_of_filters, kernel_size)
print("Model")
print(model)
model.forward(mfccs_tensor, number_of_networks)

