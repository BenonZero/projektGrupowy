import os
import random

import torch
from torch import nn, optim
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
from tqdm import tqdm

from thirdMethodModel import DeepTemplateMatchingModule

# print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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

# labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy',
          'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
          'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


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
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


class prev:
    prev_tensor = [None] * 35


IMP = prev()


def collate_fn_3r(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    # prev_tensor = [None] * 35
    tested_tensors, template_tensors, targets = [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, sample_rate, label, *_ in batch:
        is_to_be_correct = True
        if IMP.prev_tensor[labels.index(label)] is None:
            match = waveform
            IMP.prev_tensor[labels.index(label)] = waveform
        else:
            match = IMP.prev_tensor[labels.index(label)]
            IMP.prev_tensor[labels.index(label)] = waveform
            r = random.uniform(0, 1)
            if r < 0.5:
                is_to_be_correct = False
                temp = random.choice(IMP.prev_tensor)
                if temp is None or temp is match:
                    is_to_be_correct = True
                    match = IMP.prev_tensor[labels.index(label)]
                else:
                    match = temp
        new_sample_rate = 8000
        transform = transforms.Resample(
            orig_freq=sample_rate,
            new_freq=new_sample_rate)
        resampled_waveform = transform(waveform)
        resampled_match = transform(match)
        transform = transforms.MelSpectrogram(sample_rate=new_sample_rate,
                                              n_fft=400,
                                              f_min=0,
                                              f_max=None,
                                              pad=0,
                                              n_mels=128,
                                              power=2,
                                              center=True,
                                              pad_mode='constant',
                                              norm=None
                                              )
        mel_waveform = torch.squeeze(transform(resampled_waveform))
        mel_match = torch.squeeze(transform(resampled_match))

        tested_tensors += [mel_waveform]
        template_tensors += [mel_match]
        if is_to_be_correct:
            targets += [torch.tensor(1)]
        else:
            targets += [torch.tensor(0)]
    # Group the list of tensors into a batched tensor
    tested_tensors = pad_sequence(tested_tensors)
    template_tensors = pad_sequence(template_tensors)
    targets = torch.stack(targets)

    return tested_tensors, template_tensors, targets


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
    collate_fn=collate_fn_3r,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_3r,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
model = DeepTemplateMatchingModule()

optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.005, weight_decay=0.00005)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# model_3drMeth_ep4
# loading

# checkpoint = torch.load("model_3drMeth_ep4.pt", map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# curr_epoch = checkpoint['epoch']
# loss = checkpoint['loss']


def train(model, epoch, log_interval):
    model.train()
    loss = 0
    # ctr = 0

    for batch_idx, (tested, templates, target) in enumerate(train_loader):
        # ctr += 1

        # data = data.to(device)
        # target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(tested, templates)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(tested)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "model_3drMeth_ep" + str(epoch) + ".pt")


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def number_of_correct_pos(pred, target):
    # count number of correct predictions
    count_pos = 0
    tot_pos = 0
    count_neg = 0
    tot_neg = 0
    for i in range(pred.size(0)):
        if target[i] == 1:
            tot_pos += 1
        else:
            tot_neg += 1
        if pred[i] == target[i] and target[i] == 1:
            count_pos += 1
        if pred[i] == target[i] and target[i] == 0:
            count_neg += 1
    return count_pos/tot_pos, count_neg/tot_neg


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    correct_pos, correct_neg = 0, 0
    for evaluation, template, target in test_loader:
        # data = data.to(device)
        # target = target.to(device)

        output = model(evaluation, template)

        pred = get_likely_index(output)
        # assume output between 1 - 0 correct is > 80% yes

        # correct += number_of_correct(pred, target)
        correct_pos, correct_neg = number_of_correct_pos(pred, target)
        # correct = correct_neg + correct_neg
        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct_pos} , {correct_neg}")


log_interval = 30
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()
