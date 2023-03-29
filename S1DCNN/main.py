from glob import glob
import torch
import matplotlib
import librosa.display
import matplotlib.pyplot as plt
import torch.nn as nn
# Wejściem jest 13 wektorów MFCC o długości T (liczba próbek w badanym pliku dźwiękowym)

audio_files = glob('./input/example.wav')

y, sr = librosa.load(audio_files[0])                                    # load .wav file

hop_length_samples = librosa.time_to_samples(0.01)                      # liczba próbek odpowiadająca 10ms, mfcc shift
frame_length_samples = librosa.time_to_samples(0.025)                   # liczba próbek odpowiadająca 25ms, mfcc window

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length_samples,
                             win_length=frame_length_samples)           # two dimensional feature (MFCC) array , 13 features

#fig, ax = plt.subplots()
#img = librosa.display.specshow(mfccs, x_axis='off', ax=ax, hop_length=hop_length_samples,
#                               win_length=frame_length_samples)
#fig.colorbar(img, ax=[ax])
#ax.set(title='MFCC', ylabel='feature dimension', xlabel='sample')
#plt.show()

# hop wynosi 10ms, więc mamy 100 próbek na sekundę, film trwa około 34 sekundy więc powinniśmy otrzymać około 3400 próbek
# otrzymaliśmy 3370 więc input działa poprawnie
# mfccs[0] to pierwszy wektor, mfccs[1] to drugi itd.
# 0-feature na dole 12-feature na górze
                                                                #I      (13 wektorów o długości T - 3370 dla przykładowego wejścia)
mfccs_tensor = torch.tensor(mfccs)                              #Przekształcenie danych

number_of_vectors_layer_1 = 13                  #wejście warstwy 1 - 13 wektorów długości T
number_of_filters = 32                          #liczba filtrów w każdej warstwie
number_of_vectors_layer_2 = 32                  #1 wektor dla każdego filtra ^  (filtr 4 w warstwie 2 kożysta tylko z wyjścia filtra 4 w warstwie pierwszej)
number_of_logits=20                             #liczba logitów na wyjściu
kernel_size = 9
number_of_networks = 7

class Stacked1DCNN(nn.Module):
    def __init__(self):
        super(Stacked1DCNN, self).__init__()
        self.model=nn.Sequential(
        nn.Conv1d(in_channels=number_of_vectors_layer_1, out_channels=number_of_filters, kernel_size=kernel_size),      #layer1
        nn.ReLU(),
        nn.Conv1d(in_channels=number_of_vectors_layer_2, out_channels=number_of_filters, kernel_size=kernel_size)     #layer2
        )
        self.fc=nn.LazyLinear(1)
        self.softmax=nn.Softmax(dim=1)
        self.flatten=nn.Flatten(start_dim=1)
    def forward(self, input):
        networks=[]
        for i in range(number_of_networks):
            networks.append(self.model(input))

        stacked=torch.stack(networks,1)
        stacked=self.flatten(stacked)
        stacked=self.fc(stacked)

        print(len(stacked))
        print(len(stacked[0]))
        print(stacked)

        return stacked

model = Stacked1DCNN()
print("Model")
print(model)
model.forward(mfccs_tensor)

