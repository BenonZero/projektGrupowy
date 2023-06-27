from datetime import datetime
import math
import os
import random
import time
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
from SED2 import EncoderRNN, AttnDecoderRNN


SOS_token = 0
EOS_token = 1

print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # istotne dla nas -> tworzy tensory z waveformow
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)

    return batch.permute(0, 2, 1)


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
        super().__init__("../../", download=True)

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


# TODO: maxlength -> length of próbka


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # encoder_hidden = encoder.initHidden()
    encoder_hidden = None

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = input_length
    # target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        input_length, encoder.hidden_size, device=device)

    loss = 0

    # ENCODER
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # DECODER
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
    # while True:
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            timestamp = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
            model_path = f'SED_save/model_{timestamp}'
            torch.save(encoder.state_dict(), model_path + '_encoder')
            torch.save(encoder.state_dict(), model_path + '_decoder')

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


plt.switch_backend('agg')


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# TODO: ?? pairs = prepareData() czy collate()

# Create training and testing split of the data. We do not use validation in this tutorial.
print(f"[{datetime.now().strftime('%H:%M:%S')}] before generating sets")
train_set = SubsetSC("training")
print(f"[{datetime.now().strftime('%H:%M:%S')}] while generating sets")
test_set = SubsetSC("testing")
# validation_set = SubsetSC("validation")
# exit()
print(f"[{datetime.now().strftime('%H:%M:%S')}] after generating sets")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels = []
try:
    with open("../../labels.txt") as file:
        labels = file.read().split()
except:
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    with open("../../labels.txt", "w") as file:
        for l in labels:
            file.write(f"{l}\n")

print(f"[{datetime.now().strftime('%H:%M:%S')}] after sorting labels")

#  we downsample the audio for faster processing without losing too much of the classification power
# new_sample_rate = 8000
new_sample_rate = 256
transform = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

print(f"[{datetime.now().strftime('%H:%M:%S')}] before creating pairs")

waveforms = [transform(waveform) for waveform, *_ in train_set]
waveforms = pad_sequence(waveforms)

labels_index = [label_to_index(label) for _, _, label, *_ in train_set]
pairs = list(zip(waveforms, labels_index))
del waveforms
del labels_index

print(f"[{datetime.now().strftime('%H:%M:%S')}] after creating pairs")

# We don’t need to apply other transformations here

batch_size = 12

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


hidden_size = 256

encoder = EncoderRNN(256, hidden_size, device).to(device)
decoder = AttnDecoderRNN(hidden_size, 256, device).to(device)

trainIters(encoder, decoder, 5, 1, 6)
# train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

# TODO: enkoder, dekoder
# n = count_parameters(model)
# print("Number of parameters: %s" % n)


# TODO: zapisywansko
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# model_path = f'SED_save/model{}/model_{}_{}' #.format(args[0], timestamp, epoch)
# torch.save(model.state_dict(), model_path)
