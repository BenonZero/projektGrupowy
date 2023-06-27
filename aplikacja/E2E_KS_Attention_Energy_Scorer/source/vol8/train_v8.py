import sys
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim, save
import torch.utils.data as data
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
# from os import walk
from tqdm import tqdm

# from E2E_KS_Attention_Energy_Scorer.source.vol8.SED_v8 import SpeechAttDecVol8, SpeechEncVol8


torch.autograd.set_detect_anomaly(True)

###########################################################################################
# SEGMENT https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
###########################################################################################


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000),
    torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(
            labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model_encoder, model_decoder, device, train_loader, criterion, optimizer_enc, optimizer_dec, scheduler_enc, scheduler_dec, epoch_iter, iter_meter, loss, best_loss):
    model_encoder.train()
    model_decoder.train()

    model_encoder.to(device)
    model_decoder.to(device)

    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        output, hidden = model_encoder(spectrograms)
        output = model_decoder(output, hidden)

        output = output.transpose(0, 1)  # (time, batch, n_class)

        local_loss = criterion(output, labels, input_lengths, label_lengths)
        local_loss.backward()

        torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 0.20)
        torch.nn.utils.clip_grad_norm_(model_decoder.parameters(), 0.20)

        optimizer_enc.step()
        optimizer_dec.step()

        scheduler_enc.step()
        scheduler_dec.step()

        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_iter, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), local_loss.item()))

        loss[-1] = local_loss.item()
        if loss[-1] < best_loss[-1]:
            best_loss[-1] = loss[-1]
            model_path = f'model_mfcc_best_v8_e{epoch_iter}_l{int(100*loss[-1])}'
            save({
                'epoch': epoch_iter + 1,
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
                'optimizer_enc_state_dict': optimizer_enc.state_dict(),
                'optimizer_dec_state_dict': optimizer_dec.state_dict(),
                'scheduler_enc_state_dict': scheduler_enc.state_dict(),
                'scheduler_dec_state_dict': scheduler_dec.state_dict(),
                'loss': loss[-1],
                'best_loss': best_loss[-1]
            }, model_path)


def test(model_encoder, model_decoder, device, test_loader, criterion, epoch_iter, iter_meter):
    print('\nevaluating…')
    model_encoder.eval()
    model_decoder.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output, hidden = model_encoder(spectrograms)
            output = model_decoder(output, hidden)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(
                output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(
        test_loss, avg_cer, avg_wer))


def main(learning_rate=1e-5, batch_size=20, epochs=20,
         train_url="train-clean-100", test_url="test-clean"):

    hparams = {
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    torch.manual_seed(7)
    print(torch.version.cuda)
    use_cuda = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    print(device)

    if not os.path.isdir("./data"):
        os.makedirs("./data")
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(
                                       x, 'train'),
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(
                                      x, 'valid'),
                                  **kwargs)

    # model = SpeechRecognitionModel(
    #     hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
    #     hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    #     ).to(device)
    PATH = "/content/model_mfcc_v8_e9_l138"
    model_encoder = SpeechEncVol8(device, 256, n_mels=128, gru_n_layers=16, dropout=hparams['dropout'])
    model_decoder = SpeechAttDecVol8(device, 256, gru_n_layers=16, dropout=hparams['dropout'])

    print(model_encoder)
    print(model_decoder)
    print('Num Encoder Parameters', sum(
        [param.nelement() for param in model_encoder.parameters()]))
    print('Num Decoder Parameters', sum(
        [param.nelement() for param in model_decoder.parameters()]))

    optimizer_enc = optim.AdamW(
        model_encoder.parameters(), hparams['learning_rate'], eps=1e-5)
    optimizer_dec = optim.AdamW(
        model_decoder.parameters(), hparams['learning_rate'], eps=1e-5)

    # scheduler_enc = optim.lr_scheduler.OneCycleLR(optimizer_enc,
    #                                               max_lr=hparams['learning_rate'],
    #                                               steps_per_epoch=int(
    #                                                   len(train_loader)),
    #                                               epochs=hparams['epochs'],
    #                                               anneal_strategy='linear')
    # scheduler_dec = optim.lr_scheduler.OneCycleLR(optimizer_dec,
    #                                               max_lr=hparams['learning_rate'],
    #                                               steps_per_epoch=int(
    #                                                   len(train_loader)),
    #                                               epochs=hparams['epochs'],
    #                                               anneal_strategy='linear')
    scheduler_enc = optim.lr_scheduler.StepLR(
    optimizer_enc, step_size=20, gamma=0.1)
    scheduler_dec = optim.lr_scheduler.StepLR(  
    optimizer_dec, step_size=20, gamma=0.1)

    best_loss = [float('inf')]
    loss = [float('inf')]
    epoch = 0

    try:
        checkpoint = torch.load(PATH, map_location=device)
        model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
        model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
        optimizer_enc.load_state_dict(checkpoint['optimizer_enc_state_dict'])
        optimizer_dec.load_state_dict(checkpoint['optimizer_dec_state_dict'])
        scheduler_enc.load_state_dict(checkpoint['scheduler_enc_state_dict'])
        scheduler_dec.load_state_dict(checkpoint['scheduler_dec_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = [checkpoint['loss']]
        best_loss = [checkpoint['best_loss']]
        print("loaded the model successfully")
    except:
        pass
    # model.load_state_dict(torch.load("model_18_06_2023-15_37_08"))

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
                            subparam._grad.data = subparam._grad.data.to(
                                device)

    for optimizer in [optimizer_enc, optimizer_dec]:
        optimizer_to(optimizer, device)

    criterion = nn.CTCLoss(blank=28).to(device)

    iter_meter = IterMeter()
    for epoch_iter in range(epoch + 1, epochs + 1):
        train(model_encoder, model_decoder, device, train_loader, criterion, optimizer_enc,
              optimizer_dec, scheduler_enc, scheduler_dec, epoch_iter, iter_meter, loss, best_loss)
        test(model_encoder, model_decoder, device,
             test_loader, criterion, epoch_iter, iter_meter)
        model_path = f'model_mfcc_v8_e{epoch_iter}_l{int(100 * loss[-1])}'
        # save(model.state_dict(), model_path)
        save({
            'epoch': epoch_iter + 1,
            'model_encoder_state_dict': model_encoder.state_dict(),
            'model_decoder_state_dict': model_decoder.state_dict(),
            'optimizer_enc_state_dict': optimizer_enc.state_dict(),
            'optimizer_dec_state_dict': optimizer_dec.state_dict(),
            'scheduler_enc_state_dict': scheduler_enc.state_dict(),
            'scheduler_dec_state_dict': scheduler_dec.state_dict(),
            'loss': loss[-1],
            'best_loss': best_loss[-1]
        }, model_path)
###########################################################################################
# END SEGMENT
###########################################################################################

main()