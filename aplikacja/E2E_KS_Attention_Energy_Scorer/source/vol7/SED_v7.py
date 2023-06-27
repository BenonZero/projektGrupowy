import numpy as np
from torch import bmm, from_numpy, nn, transpose, cat, exp, sum, vstack, zeros
import torch.nn.functional as F


class SpeechEncVol7(nn.Module):
    def __init__(self, device, hidden_size, n_mfcc=13, dropout=0.2):
        """
        1-layer bidirectional GRU
        conv_kernel_size was 1
        hidden_size = 256 // mozna zmniejszyc
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.conv_kernel_size = 3
        self.max_pool_stride = 2
        self.gru_n_layers = 1
        self.dropout = dropout

        self.add_module("dropout1", nn.Dropout(self.dropout))
        self.add_module("first_cnn", nn.Conv1d(
            in_channels=n_mfcc, out_channels=self.hidden_size // 2, kernel_size=self.conv_kernel_size))
        # "Number of filters = Number of out_channels."
        # "Kernel size of 3 works fine everywhere"
        self.add_module("dropout2", nn.Dropout(self.dropout))
        self.add_module("second_cnn", nn.Conv1d(
            in_channels=self.hidden_size // 2, out_channels=self.hidden_size, kernel_size=self.conv_kernel_size))
        self.add_module("pool", nn.MaxPool1d(
            kernel_size=1, stride=self.max_pool_stride))
        self.add_module("dropout3", nn.Dropout(self.dropout))
        self.add_module("gru_encoder", nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.gru_n_layers, bidirectional=True))
        self.add_module("2h_to_h", nn.Linear(
            2 * self.hidden_size, self.hidden_size))
        self.add_module("encoder_relu", nn.ReLU())

    def forward(self, X):
        """
        N - batch_size, L - sequence_length, h - hidden_size (= GRU_input_size)
        input(X): (N, n_MFCC, L) // n_MFCC = 13 by default
        output: (L, N, h)
        """
        modules = self.named_modules()
        d = dict(modules)
        batch_size = X.size(0)

        # some MFCC inputs come in shape of (N, 1, 13, L)
        X = X.squeeze()
        # (N, 13, L)

        # input is already packed in the collate_fn
        # X = pack_padded_sequence(X.transpose(1, 2), [xi.size(2) for xi in X], batch_first=True, enforce_sorted=False, padding_value=0.)
        X = d["dropout1"](X)
        X = (d["first_cnn"])(X)
        # (N, h/2, L)

        X = d["dropout2"](X)
        X = (d["second_cnn"])(X)
        # (N, h, L)

        X = (d["pool"])(X)
        # (N, h_in, L) // L = L/2

        X = transpose(X, 1, 2)
        # (N, L, h_in)
        X = transpose(X, 0, 1)
        # (L, N, h_in)

        # gru wants: L,N,h_in <- sequence_length, batch, hidden_size
        # hidden: Layers(2), N, h_out // 2 bc bidirectional

        X = d["dropout3"](X)
        # initial hidden is zeros by default
        X, hidden = d["gru_encoder"](X)
        # (L, N, 2h), (Layers(2), N, h)

        # enc_out = enc_out_forward + enc_out_backward
        # X = sum(X.view(X.size(0), batch_size, 2, self.hidden_size), dim=2)

        X = X.view(X.size(0), batch_size, 2, self.hidden_size)[:, :, 0] + from_numpy(np.flip(X.view(X.size(
            0), batch_size, 2, self.hidden_size)[:, :, 1].detach().cpu().numpy(), axis=-1).copy()).to(self.device)
        X /= 2

        # X = cat((, ), dim=-1)
        # X = d["2h_to_h"](X)
        # X = X[:, :, :self.hidden_size]

        X = d["encoder_relu"](X)
        # L, N, h_out

        # we ignore the first hidden state of bidir-GRU
        # alternatives (all seem legit):
        # concatenation, sum
        last_layer = -1

        return (X, hidden[last_layer].unsqueeze(0))


class SpeechAttDecVol7(nn.Module):
    def __init__(self, device, hidden_size, output_size=10, dropout=0.2):
        """
        hidden_size = 256
        fixed output size of 10 characters
        """
        super().__init__()
        # depends on the encoder behaviour - hard to set dynamically
        self.number_of_characters = 27  # 26 + pad_value(26)
        self.sequence_length = 39
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_n_layers = 1
        self.device = device
        self.dropout = dropout

        self.add_module("dropout1", nn.Dropout(self.dropout))
        self.add_module("fc_cat_to_h", nn.Linear(
            2 * self.hidden_size, self.hidden_size))

        self.add_module("gru_decoder", nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.gru_n_layers))

        self.add_module("dropout2", nn.Dropout(self.dropout))

        self.add_module("fc_to_output", nn.Linear(
            in_features=self.hidden_size, out_features=27))

        self.add_module("dropout3", nn.Dropout(self.dropout))

        self.add_module("output_quantization", nn.Linear(
            in_features=self.sequence_length, out_features=self.output_size))  # , bias=False

    def get_idxs(self, seq_iter, len_of_seq):
        from_idx, to_idx = 0, 0
        size_of_window = len_of_seq // 5
        if seq_iter <= size_of_window // 2:
            from_idx = 0
        else:
            from_idx = seq_iter - size_of_window // 2
        if len_of_seq - seq_iter <= size_of_window // 2:
            to_idx = len_of_seq - 1
            from_idx = len_of_seq - size_of_window - 1
        else:
            to_idx = from_idx + size_of_window
        return from_idx, to_idx

    def att(self, enc_out, dec_h):
        """
        enc_out:  (N, L, h)
        dec_h:    (N, 1, h)
        returns attention weights: (N, L)
        czy softmax na koncu?
        """
        # bmm: (N,1,h)*(N,h,L) -> (N, 1, L) -> (N,L)
        # bmm: (N,L,h)*(N,h,L) -> (N, L, L) -> (N,L)
        # (N,L)/(N,L)

        # print(f"\nbefore bmm:\n\tdec_h({dec_h.min()},{dec_h.max()})\n\tenc_out({enc_out.min()},{enc_out.max()})")

        score = bmm(dec_h, enc_out.transpose(1, 2)).squeeze()
        scores = bmm(enc_out, dec_h.transpose(
            1, 2).repeat((1, 1, enc_out.size(1))))

        # print(f"\nbefore exp:\n\tscore({score.min()},{score.max()})\n\tscores({scores.min()},{scores.max()})")
        # some max values are huge fsr
        # there are even infs...

        score = exp(score)
        scores = exp(scores)

        if score.isinf().any() or scores.isinf().any():
            print("attention please: exp -> inf values >_<")
        # print(f"\after exp:\n\tscore({score.min()},{score.max()})\n\tscores({scores.min()},{scores.max()})")

        ret = F.softmax(score /
                        sum(scores, dim=2), dim=-1)
        # ret = exp(bmm(dec_h, enc_out.transpose(1, 2)).squeeze(
        # )) / sum(exp(bmm(enc_out, dec_h.transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2)
        # print(ret.size()) # (N,L) confirmed
        return ret
        # (N, L)

    def forward(self, gru_encoder_output, hidden):
        """
        gru_encoder_output: (L, N, h)
        hidden: (1, N, h)
        """
        d = dict(self.named_modules())

        gru_encoder_output = d["dropout1"](gru_encoder_output)
        decoder_outputs = None
        for seq_iter in range(gru_encoder_output.size(0)):
            # cat
            # bmm: (N, 1, L)*(N, L, h) -> (N, 1, h)

            # from_idx, to_idx = self.get_idxs(
            #     seq_iter, gru_encoder_output.size(0))
            # decoder_output, hidden = d["gru_decoder"](
            #     d["fc_cat_to_h"](cat((hidden,
            #                           bmm(self.att(gru_encoder_output[from_idx:to_idx].transpose(0, 1), hidden.transpose(0, 1)).unsqueeze(1),
            #                               gru_encoder_output[from_idx:to_idx].transpose(0, 1)).transpose(0, 1)), dim=-1)), hidden)

            decoder_output, hidden = d["gru_decoder"](
                d["fc_cat_to_h"](cat((hidden,
                                      bmm(self.att(gru_encoder_output.transpose(0, 1), hidden.transpose(0, 1)).unsqueeze(1),
                                          gru_encoder_output.transpose(0, 1)).transpose(0, 1)), dim=-1)), hidden)

            # decoder_input = gru_encoder_output.unbind(
            #     dim=0)[seq_iter].unsqueeze(0).clone()

            # decoder_output, hidden = d["gru_decoder"](decoder_input, hidden)
            # decoder_output = d["fc_cat_to_h"](cat((decoder_output,
            #                                        bmm(self.att(gru_encoder_output.transpose(0, 1), decoder_output.transpose(0, 1)).unsqueeze(1),
            #                                            gru_encoder_output.transpose(0, 1)).transpose(0, 1)), dim=-1))

            if decoder_outputs is None:
                decoder_outputs = decoder_output
                # (1, N, h_out)
            else:
                decoder_outputs = vstack((decoder_outputs, (decoder_output)))
                # (<2;L>, N, h_out)

            # if decoder_output.equal(EOS):
            #     break

        decoder_outputs = d["dropout2"](decoder_outputs)
        # (L, N, h_out)
        decoder_outputs = d["fc_to_output"](decoder_outputs)
        # (L, N, 28)
        decoder_outputs = decoder_outputs.transpose(0, 1)
        # (N, L, 28)

        # dobrze by bylo cos z tym zrobic
        decoder_outputs = d["dropout3"](decoder_outputs)
        decoder_outputs = d["output_quantization"](
            decoder_outputs.transpose(1, 2)).transpose(1, 2)

        decoder_outputs = decoder_outputs.clamp(min=1e-4)
        if decoder_outputs.isnan().any():
            print("\nisNan:\n" + str(decoder_outputs.isnan().any()))
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs
