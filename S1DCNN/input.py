import librosa
from glob import glob
import librosa.display
import matplotlib.pyplot as plt

#Wejściem jest 13 wektorów MFCC o długości T (liczba próbek w badanym pliku dźwiękowym)

audio_files = glob('./input/*.wav')

y, sr = librosa.load(audio_files[0])                          #load .wav file

hop_length_samples = librosa.time_to_samples(0.01)      #liczba próbek odpowiadająca 10ms, mfcc shift
frame_length_samples = librosa.time_to_samples(0.025)   #liczba próbek odpowiadająca 25ms, mfcc window

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length_samples, win_length=frame_length_samples)         #two dimensional feature (MFCC) array , 13 features


fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='off', ax=ax, hop_length=hop_length_samples, win_length=frame_length_samples)
fig.colorbar(img, ax=[ax])
ax.set(title='MFCC', ylabel='feature dimension', xlabel='sample')
plt.show()

#hop wynosi 10ms, więc mamy 100 próbek na sekundę, film trwa około 34 sekundy więc powinniśmy otrzymać około 3400 próbek
#otrzymaliśmy 3370 więc input działa poprawnie
print(len(mfccs[0]))
print(mfccs[0])                                 #mfccs[0] to pierwszy wektor, mfccs[1] to drugi itd.
                                                #0-feature na dole 12-feature na górze