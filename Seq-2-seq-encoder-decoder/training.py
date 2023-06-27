import sys
from datetime import datetime
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, optim, save
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
from tqdm import tqdm
import random


PADDING_CHAR = "{"
PADDING_INDEX = 26
NO_PAD = 0
PAD_SYMM = 1
PAD_BACK = 2


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


# def count_parameters(model):
#     return torch.sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def label_to_target_tensor(s, tg_size=10, mode=PAD_SYMM):
    """
    applies padding and returns a tensor of indices
    """
    ret = torch.tensor(list(map(lambda c: ord(c) - 97, s.lower())), device=device)
    if mode == PAD_BACK:
        padding = tg_size - len(s)
        ret = F.pad(ret, (0, padding), value=PADDING_INDEX)
    elif mode == PAD_SYMM:
        rest = padding % 2
        padding //= 2
        ret = F.pad(ret, (padding, padding + rest), value=PADDING_INDEX)
    return ret

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


def train(model_encoder, model_decoder, epoch_parameter, log_interval):
    model_encoder.train()
    model_decoder.train()

    model_encoder.to(device)
    model_decoder.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform_resample(data).to(device)
        data = transform_mfcc(data).to(device)
        # print(f"shape of mfcc: {tuple(data.size())}")

        # for 2nd method - SED output is a string of length 50

        (enc_output, enc_hidden) = model_encoder(data)
        output = model_decoder(enc_output, enc_hidden)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        # print(f"output, target: {tuple(output.size())}, {tuple(stacked_targets.size())}")
        # output: (N, output_len, n_classes), targets: (N, output_len)

        # ############################################################################################
        # # STATIC PADDING
        # #
        # stacked_targets = torch.stack([label_to_target_tensor(index_to_label(iter_target), output.size(1)) for iter_target in target])
        # stacked_targets = torch.tensor(stacked_targets.tolist(), dtype=torch.long, device=device)
        # loss = F.nll_loss(output.transpose(-1,-2), stacked_targets)
        # # loss = F.nll_loss(output, torch.stack([label_to_target_tensor(index_to_label(iter_target)) for iter_target in target]))
        # #
        # ############################################################################################

        ############################################################################################
        # DYNAMIC PADDING
        #
        # print("\n" + str(output.argmax(dim=-1)[0]) + " ->")
        #
        dynamic_output = []
        for batch_iter in range(output.size(0)):
            # (L, C)
            pred = output[batch_iter].argmax(dim=-1)
            # (L)
            shift = 0
            while shift < pred.size(0) and pred[shift].item() == PADDING_INDEX:
                shift += 1
            dynamic_output.append(output[batch_iter].roll(-shift, 0))
        dynamic_output = torch.stack(dynamic_output) # (N, L, C)
        #
        # print("\n" + str(dynamic_output.argmax(dim=-1)[0]))
        #
        stacked_targets = torch.stack([label_to_target_tensor(index_to_label(iter_target), mode=PAD_BACK) for iter_target in target])
        stacked_targets = torch.tensor(stacked_targets.tolist(), dtype=torch.long, device=device)
        loss = F.nll_loss(dynamic_output.transpose(-1,-2), stacked_targets)
        #
        ############################################################################################

        # # ############################################################################################
        # # nn.CTCLoss
        # #
        # nn.CTCLoss
        # #
        # # ############################################################################################

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        loss.backward()

        #prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 0.25)
        torch.nn.utils.clip_grad_norm_(model_decoder.parameters(), 0.25)

        optimizer_enc.step()
        optimizer_dec.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch_parameter} [{batch_idx * len(data)}/{len(train_loader.dataset):.4f} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
        if losses[-1] < best_loss[-1]:
            os.system("rm *model_mfcc_best*")
            best_loss[-1] = losses[-1]
            model_path = f'GUR4_model_mfcc_best_v3_e{epoch_parameter}_l{int(100*losses[-1])}'
            # save(model.state_dict(), model_path)
            save({
                'epoch': epoch_parameter,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_enc_state_dict': optimizer_enc.state_dict(),
                'optimizer_dec_state_dict': optimizer_dec.state_dict(),
                'scheduler_enc_state_dict': scheduler_enc.state_dict(),
                'scheduler_dec_state_dict': scheduler_dec.state_dict(),
                'loss': losses[-1],
                'best_loss': best_loss[-1]
                }, model_path)


def test(model_encoder, model_decoder, epoch):
    TEST_SIZE = 50
    progress = tqdm(range(TEST_SIZE), f"Testing epoch: {epoch}")
    invalid_num = 0
    good_guess = 0
    waveform_len, *_ = test_set[0]
    waveform_len = transform_mfcc(waveform_len.to(device))
    waveform_len = waveform_len.size(2)

    model_encoder.eval()
    model_decoder.eval()

    model_encoder.to(device)
    model_decoder.to(device)

    to_file = ""
    for i in range(TEST_SIZE):
        choice = random.choice(test_set)
        local_waveform, _, label, *_ = choice
        local_waveform = transform_mfcc(local_waveform.to(device)).to(device)

        get, rid_of, invalids = 0, 0, 0
        padding_value = local_waveform[get][rid_of][invalids]
        padding = abs(waveform_len - local_waveform.size(2))
        rest = padding % 2
        padding //= 2
        local_waveform = F.pad(local_waveform, (padding, padding + rest), value=padding_value)

        local_waveform = local_waveform.repeat((2, 1, 1))

        try:
            progress.update()
            (enc_output, enc_hidden) = model_encoder(local_waveform)
            output = model_decoder(enc_output, enc_hidden)
        except Exception as e:
            # print(f"Invalid input: LABEL={label} ERROR={e}")
            invalid_num += 1
            continue

        l = output[0].tolist()
        s = ""
        for c in l:
            s += (chr(c.index(max(c)) + 97))
        s = s.strip(PADDING_CHAR)
        to_file += f"{label} -> {s} " + \
                   f"{'Same!' if s == label else ''}\n"
        if s == label:
            good_guess += 1
        try:
            acc = int((good_guess / (i-invalid_num)) * 100)
        except ZeroDivisionError:
            acc = 0
        stats = f"Testing epoch: {epoch} | "\
                f"Current accuracy: {acc}% | "\
                f"Good: {good_guess} | "\
                f"Invalid: {invalid_num} | "\
                f"Iterations: {i+1}"
        progress.set_description(stats)
        with open(f"{epoch}_v3_acc.txt", "w") as file:
            file.write(stats + "\n" + to_file)

    progress.close()

print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

torch.autograd.set_detect_anomaly(True)

PATH = "/content/gg2_model_mfcc_best_v3_e16_l10"
# def __init__(self, device, dec_h_size=256, seq_len=128, input_size=256, output_size=10) -> None:
#  (self, device, n_mfcc=13, input_size=256, enc_h_size=256)
model_encoder = SpeechEncVol3(device, hidden_size=256, gru_layers=2)
model_decoder = SpeechAttDecVol3(device, hidden_size=256, gru_layers=2)
optimizer_enc = optim.Adam(model_encoder.parameters(), lr=0.01, weight_decay=0.0001)
optimizer_dec = optim.Adam(model_decoder.parameters(), lr=0.01, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler_enc = optim.lr_scheduler.StepLR(optimizer_enc, step_size=20, gamma=0.1)
scheduler_dec = optim.lr_scheduler.StepLR(optimizer_dec, step_size=20, gamma=0.1)
best_loss = [float('inf')]
try:
    checkpoint = torch.load(PATH)
    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
    optimizer_enc.load_state_dict(checkpoint['optimizer_enc_state_dict'])
    optimizer_dec.load_state_dict(checkpoint['optimizer_dec_state_dict'])
    scheduler_enc.load_state_dict(checkpoint['scheduler_enc_state_dict'])
    scheduler_dec.load_state_dict(checkpoint['scheduler_dec_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_loss = [checkpoint['best_loss']]
    best_loss = [loss]
    print("loaded the model successfully")
except:
    epoch = 0

print(model_encoder)
print(model_decoder)
print('Num Encoder Parameters', sum(
    [param.nelement() for param in model_encoder.parameters()]))
print('Num Decoder Parameters', sum(
    [param.nelement() for param in model_decoder.parameters()]))

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

for optimizer in [optimizer_enc, optimizer_dec]:
    optimizer_to(optimizer, device)
# model.load_state_dict(torch.load("model_18_06_2023-15_37_08"))

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
# new_sample_rate = 256
# transform_resample = torchaudio.transforms.Resample(
#     orig_freq=sample_rate, new_freq=new_sample_rate)
# transform_mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 80, "n_mels": 23, "center": False}) # n_mfcc=13 (default:40), , log_mels=True
transform_mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13) # n_mfcc=13 (default:40), , log_mels=True

# transformed = transform(waveform)
# We don’t need to apply other transformations here
# print("after resampling")

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

# loss_fn = nn.CrossEntropyLoss()
# loss_fn = F.nll_loss
# loss_fn = nn.NLLLoss



log_interval = 20
n_epoch = 25

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []
model_path = "last_model"

# The transform needs to live on the same device as the model and the data.

# transform = transform.to(device)
# transform_resample = transform_resample.to(device)
transform_mfcc = transform_mfcc.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch_iter in range(epoch + 1, epoch + n_epoch + 1):
        train(model_encoder, model_decoder, epoch_iter, log_interval)
        timestamp = datetime.now().strftime('%m_%Y-%H_%M')
        model_path = f'GUR4_model_mfcc_v3_e{epoch_iter}_l{int(100*losses[-1])}'
        # save(model.state_dict(), model_path)
        scheduler_enc.step()
        scheduler_dec.step()
        save({
            'epoch': epoch_iter,
            'model_encoder_state_dict': model_encoder.state_dict(),
            'model_decoder_state_dict': model_decoder.state_dict(),
            'optimizer_enc_state_dict': optimizer_enc.state_dict(),
            'optimizer_dec_state_dict': optimizer_dec.state_dict(),
            'scheduler_enc_state_dict': scheduler_enc.state_dict(),
            'scheduler_dec_state_dict': scheduler_dec.state_dict(),
            'loss': losses[-1],
            'best_loss': best_loss[-1]
            }, model_path)
        test(model_encoder, model_decoder, epoch_iter)
