# TODO https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import sys
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, optim, save
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
# from os import walk
from tqdm import tqdm

from functions import *
from E2E_KS_Attention_Energy_Scorer.source.SED import SpeechEncDec
from E2E_KS_Attention_Energy_Scorer.source.QED import QueryEncDec
from E2E_KS_Attention_Energy_Scorer.source.Att import AttMech
from E2E_KS_Attention_Energy_Scorer.source.Energy import EneSc
# import your models here

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """KOLACJONOWANIE DANYCH"""
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

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
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# depracated
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    model.train()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    # "C:\\Users\\kubak\\INF_SEM5\\PG\\repo\\projektGrupowy\\speech_commands_v0.02"

    training_loader = []
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        inputs = "bird"
        target = inputs.lower()
        target = [float(ord(i)) for i in target]
        target = [target[i % len(target)] for i in range(256)]
        target = tensor([target])

        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(model, epoch, log_interval):
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)

        # for 2nd method - SED output is a string of length 50
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

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


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        # print("data: " + data + "target " + target)
        # exit()
        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


# roznice w datasetach
# roznice w modelach
# run the script with $ py training.py [model_nr]
args = sys.argv[1:]
if len(args) < 1:
    exit()

model = None
if args[0] == "1":
    # obsluga metody 1.
    model = "obsluga metody 1"
elif args[0] == "2":
    # obsluga metody 2.
    if len(args) < 2:
        print("\033[91m{}\033[00m".format("[indian accent] sorry sir but your input is INCORRECT"))
        exit()        
    elif args[1] == "SED":
        model = SpeechEncDec()
    elif args[1] == "QED":
        model = QueryEncDec()
    elif args[1] == "ATT_ESC":
        model = AttMech()
        # TODO jeden model dwa moduly
elif args[0] == "3":
    # obsluga metody 3.
    model = "obsluga metody 3"
else:
    print("[indian accent] sorry sir but your input is INCORRECT")
    exit()

print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("training: " + str(model))

# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
print("while generating sets")
test_set = SubsetSC("testing")
# validation_set = SubsetSC("validation")
# exit()
print("after generating sets")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels = []
try:
    with open("labels.txt") as file:
        labels = file.read().split()
except:
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    with open("labels.txt", "w") as file:
        for l in labels:
            file.write(f"{l}\n")

print("after sorting labels")



#  we downsample the audio for faster processing without losing too much of the classification power
# new_sample_rate = 8000
new_sample_rate = 256
transform = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)
# We don’t need to apply other transformations here
print("after resampling")
word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)
print(word_start, "-->", index, "-->", word_recovered)

# x = input("wanna fly further? [y/n]\n")
# if x.capitalize() != 'Y':
#     exit()

batch_size = 12
# batch_size = 128

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
# validation_loader = torch.utils.data.DataLoader(
#     validation_set,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )

n = count_parameters(model)
print("Number of parameters: %s" % n)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# region old_code
# TRAINING AND TESTING NETWORK (BEJBE)

# log_interval = 20
# n_epoch = 2

# pbar_update = 1 / (len(train_loader) + len(test_loader))
# losses = []
# best_vloss = 1_000_000.
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# # The transform needs to live on the same device as the model and the data.
# transform = transform.to(device)
# with tqdm(total=n_epoch) as pbar:
#     for epoch in range(1, n_epoch + 1):
#         train(model, epoch, log_interval)
#         test(model, epoch)
#         scheduler.step()

#     # avg_vloss = running_vloss / (i + 1)
#     # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

#     # # Track best performance, and save the model's state
#     # if avg_vloss < best_vloss:
#     #     best_vloss = avg_vloss
#     #     model_path = '../model_{}_{}'.format(timestamp, epoch)
#     #     save(model.state_dict(), model_path)
# model_path = 'models/model_{}_{}'.format(timestamp, epoch)
# save(model.state_dict(), model_path)





# def predict(tensor):
#     # Use the model to predict the label of the waveform
#     tensor = tensor.to(device)
#     tensor = transform(tensor)
#     tensor = model(tensor.unsqueeze(0))
#     tensor = get_likely_index(tensor)
#     tensor = index_to_label(tensor.squeeze())
#     return tensor


# waveform, sample_rate, utterance, *_ = train_set[-1]

# print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")





# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");

# exit()

# endregion

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()
        model_path = 'models/model{}/model_{}_{}'.format(args[0], timestamp, epoch)
        save(model.state_dict(), model_path)


# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");











# Initializing in a separate cell so we can easily add more epochs to the same run
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter(f'../runs/QED_trainer_{timestamp}')
# epoch_number = 0

# EPOCHS = 5

# best_vloss = 1_000_000.

# for epoch in range(EPOCHS):
#     print('EPOCH {}:'.format(epoch_number + 1))

#     # Make sure gradient tracking is on, and do a pass over the data
#     model.train(True)
#     avg_loss = train_one_epoch(epoch_number, writer)

#     # We don't need gradients on to do reporting
#     # TODO ogarnac czy wszystko dziala z train(False) (metadane)
#     model.train(False)

#     running_vloss = 0.0

# TODO czy potrzebujemy validation_loader
#     for i, vdata in enumerate(validation_loader):
#         vinputs, vlabels = vdata
#         voutputs = model(vinputs)
#         vloss = loss_fn(voutputs, vlabels)
#         running_vloss += vloss

#     avg_vloss = running_vloss / (i + 1)
#     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

#     # Log the running loss averaged per batch
#     # for both training and validation
#     writer.add_scalars('Training vs. Validation Loss',
#                        {'Training': avg_loss, 'Validation': avg_vloss},
#                        epoch_number + 1)
#     writer.flush()

#     # Track best performance, and save the model's state
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = '../model_{}_{}'.format(timestamp, epoch_number)
#         save(model.state_dict(), model_path)

#     epoch_number += 1
