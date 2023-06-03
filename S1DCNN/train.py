
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
import torch.optim.adam
import os
import input
import modelik
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

number_of_vectors_layer_1 = 13  # wejście warstwy 1 - 13 wektorów długości T
number_of_filters = 35  # liczba filtrów w każdej warstwie
number_of_vectors_layer_2 = 35  # 1 wektor dla każdego filtra ^  (filtr 4 w warstwie 2 kożysta tylko z wyjścia filtra 4 w warstwie pierwszej)
kernel_size = 3


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

#dlugosc=len(train_set)
#train_set1=[]
#for i in range(1,dlugosc):
    #train_set1.append(input.crete_mfccs_vectors2(train_set[i][0]))
    #print(train_set[i])

#print(train_set[0][0])
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

# waveform=waveform.flatten()

#mfccs=input.crete_mfccs_vectors2(waveform)
#print(mfccs)
#model = modelik.Stacked1DCNN(number_of_vectors_layer_1, number_of_vectors_layer_2, number_of_filters, kernel_size)
#print("Model")
#print(model)
#model.forward(mfccs)
#print(model)


labels = []
try:
    with open("labels.txt") as file:
        labels = file.read().split()
except:
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    with open("labels.txt", "w") as file:
        for l in labels:
            file.write(f"{l}\n")


#transformed=input.create_mfccs_vectors3(waveform)
def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """A CO TO ROBI?"""

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number


    # Gather in lists, and encode labels as indices
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        #waveform1=waveform
        #waveform=input.create_mfccs_vectors2(waveform1)
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    targets=targets.type(torch.LongTensor)
    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
model = modelik.Stacked1DCNN(number_of_vectors_layer_1, number_of_filters,number_of_filters ,kernel_size)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data=input.create_mfccs_vectors3(data).to(device)
        # apply transform and model on whole batch directly on device
        output = model(data)
        #print(output)
        #print(target)
        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each  element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = input.create_mfccs_vectors3(data).to(device)
        output = model(data)
        #print(output)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = 20
n_epoch = 5

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
#transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()