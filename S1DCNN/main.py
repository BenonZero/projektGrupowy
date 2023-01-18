import numpy
import torch
import librosa
from glob import glob
import librosa.display
import matplotlib.pyplot as plt
import torch.nn as nn

# Wejściem jest 13 wektorów MFCC o długości T (liczba próbek w badanym pliku dźwiękowym)

audio_files = glob('./input/*.wav')

y, sr = librosa.load(audio_files[0])  # load .wav file

hop_length_samples = librosa.time_to_samples(0.01)  # liczba próbek odpowiadająca 10ms, mfcc shift
frame_length_samples = librosa.time_to_samples(0.025)  # liczba próbek odpowiadająca 25ms, mfcc window

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length_samples,
                             win_length=frame_length_samples)  # two dimensional feature (MFCC) array , 13 features

fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='off', ax=ax, hop_length=hop_length_samples,
                               win_length=frame_length_samples)
fig.colorbar(img, ax=[ax])
ax.set(title='MFCC', ylabel='feature dimension', xlabel='sample')
plt.show()

# hop wynosi 10ms, więc mamy 100 próbek na sekundę, film trwa około 34 sekundy więc powinniśmy otrzymać około 3400 próbek
# otrzymaliśmy 3370 więc input działa poprawnie
print(len(mfccs[0]))
print(len(mfccs))  # mfccs[0] to pierwszy wektor, mfccs[1] to drugi itd.
# 0-feature na dole 12-feature na górze

X = numpy.random.uniform(-10, 10, 260).reshape(1, 13, -1)
mfccs_tensor = torch.tensor(mfccs)
# print(X)
# Y = np.random.randint(0, 9, 10).reshape(1, 1, -1)

# musze jakos dopasowac to wejscie z pliku audio
kernel_size = 9
number_of_filters = 32
number_of_vectors_layer_1 = 13
number_of_vectors_layer_2 = 32
number_of_logits=20

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()

        self.layer1 = nn.Conv1d(in_channels=number_of_vectors_layer_1, out_channels=number_of_filters,
                                kernel_size=kernel_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=number_of_vectors_layer_2, out_channels=number_of_filters,
                                kernel_size=kernel_size)
        self.linear_layer1=nn.Linear(3354,number_of_logits)
        self.linear_layer2=nn.Linear(20,1)

    def forward(self, x):
        x=self.layer1(x)
        # y=self.layer1(y)
        # z=self.layer1(z)
        # a=self.layer1(a)
        x=self.act1(x)
        # y=self.act1(y)
        # z=self.act1(z)
        # a=self.act1(a)
        x=self.layer2(x)
        # y=self.layer2(y)
        # z=self.layer2(z)
        # a=self.layer2(a)
        x=self.linear_layer1(x)
        x=self.linear_layer2(x)
        print(len(x))
        print(len(x[0]))
        #log_probs = torch.nn.functional.log_softmax(x, dim=1)

        # log_probsy = torch.nn.functional.log_softmax(y, dim=1)
        # log_probsz = torch.nn.functional.log_softmax(z, dim=1)
        # log_probsa = torch.nn.functional.log_softmax(a, dim=1)
        # log_probs=(log_probsa+log_probsz+log_probsx+log_probsy)/4
        #log_probs=nn.Softmax(x)
        #print (len(log_probs))
        #print(len(log_probs[0]))
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

# class Simple1DCNN(nn.Module):
#     def __init__(self):
#         super(Simple1DCNN, self).__init__()
#         self.layers1 = nn.ModuleList([
#             nn.Conv1d(in_channels=number_of_vectors_layer_1,
#                       out_channels=number_of_filters,
#                       kernel_size=kernel_size)
#             for _ in range(7)
#         ])
#         self.acts1 = nn.ModuleList([nn.ReLU() for _ in range(7)])
#         self.layers2 = nn.ModuleList([
#             nn.Conv1d(in_channels=number_of_vectors_layer_2,
#                       out_channels=number_of_filters,
#                       kernel_size=kernel_size)
#             for _ in range(7)
#         ])
#
#     def forward(self, x):
#         for layer in self.layers1:
#             x = layer(x)
#         for act in self.acts1:
#             x = act(x)
#         for layer in self.layers2:
#             x = layer(x)
#         log_probs = torch.nn.functional.log_softmax(x, dim=1)
#         return log_probs


model = Simple1DCNN().float()
print(model)
tensor = mfccs_tensor.clone().detach()
print(model(tensor))
logit = model(tensor)
p = nn.functional.softmax(logit, dim=1)
print(p)
print(len(p))
