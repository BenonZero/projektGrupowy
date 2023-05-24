from glob import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import librosa.display
import matplotlib.pyplot as plt
import torchaudio.transforms
from torchaudio.transforms import MFCC
def create_mfccs_vectors():
    audio_files = glob('./SpeechCommands/speech_commands_v0.02/down/00b01445_nohash_0.wav')
    y, sr = librosa.load(audio_files[0])  # load .wav file
    #print(y)
    hop_length_samples = librosa.time_to_samples(0.01)  # liczba próbek odpowiadająca 10ms, mfcc shift
    frame_length_samples = librosa.time_to_samples(0.025)  # liczba próbek odpowiadająca 25ms, mfcc window
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length_samples,
                                  win_length=frame_length_samples)  # two dimensional feature (MFCC) array , 13 features

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(mfccs, x_axis='off', ax=ax, hop_length=hop_length_samples,
    #                               win_length=frame_length_samples)
    # fig.colorbar(img, ax=[ax])
    # ax.set(title='MFCC', ylabel='feature dimension', xlabel='sample')
    # plt.show()
    # hop wynosi 10ms, więc mamy 100 próbek na sekundę, film trwa około 34 sekundy więc powinniśmy otrzymać około 3400 próbek
    # otrzymaliśmy 3370 więc input działa poprawnie
    # mfccs[0] to pierwszy wektor, mfccs[1] to drugi itd.
    # 0-feature na dole 12-feature na górze
    # I      (13 wektorów o długości T - 3370 dla przykładowego wejścia)
    tensor = torch.tensor(mfccs)
    # Przekształcenie danych
    return tensor

def create_mfccs_vectors2(audio_file1):
    hop_length_samples = librosa.time_to_samples(0.01)  # liczba próbek odpowiadająca 10ms, mfcc shift
    frame_length_samples = librosa.time_to_samples(0.025)  # liczba próbek odpowiadająca 25ms, mfcc window
    transform=torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=13,melkwargs={"n_fft":1000, "hop_length": hop_length_samples, "win_length": frame_length_samples})
    tensor=transform(audio_file1)
    mfccs=tensor
    tensor=torch.tensor(mfccs)
    tensor=np.squeeze(tensor)
    return tensor

def create_mfccs_vectors3(waveform):
    sample_rate=16000
    mfcc_transform=MFCC(
        sample_rate=sample_rate,
        n_mfcc=13
    )

    mfcc=mfcc_transform(waveform)
    mfcc=np.squeeze(mfcc)
    return mfcc

