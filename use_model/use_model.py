import os
import random
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
import torch.nn.functional as F
from SpeechAttDec import SpeechAttDecVol3
from SpeechEnc import SpeechEncVol3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13)

PADDING_CHAR = "{"
PADDING_INDEX = 26


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


labels = ["backward", "bed", "bird", "cat", "dog",
          "down", "eight", "five", "follow", "forward",
          "four", "go", "happy", "house", "learn",
          "left", "marvin", "nine", "no", "off",
          "on", "one", "right", "seven", "sheila",
          "six", "stop", "three", "tree", "two",
          "up", "visual", "wow", "yes", "zero"]
test_set = SubsetSC("testing")
batch_size = 256


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


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


if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


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

PATH = "D:\Studia\pythonProjects\test_model\modele\OGUR1_TOP_MAVERICK\GUR4_model_mfcc_best_v3_e32_l1"
model_encoder = SpeechEncVol3(device,  gru_layers=2)
model_decoder = SpeechAttDecVol3(device, gru_layers=2)
checkpoint = torch.load(PATH, map_location=device)
model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
epoch = 1
test(model_encoder, model_decoder, epoch)