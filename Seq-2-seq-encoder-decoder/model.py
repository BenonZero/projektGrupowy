import torch
from torch import bmm, nn, cat, vstack, transpose, zeros
import torch.nn.functional as F


class SpeechEncVol3(nn.Module):
    def __init__(self, device, n_mel=13, hidden_size=256, gru_layers=4, dropout=0.1):
        super().__init__()

        self.device = device
        max_pool_stride = 2
        kernel_size = 3

        self.add_module("batch_norm", nn.BatchNorm1d(hidden_size // 2))
        self.add_module("first_cnn", nn.Conv1d(
            in_channels=n_mel, out_channels=hidden_size // 2, kernel_size=kernel_size))
        self.add_module("second_cnn", nn.Conv1d(
            in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=kernel_size))
        self.add_module("pool", nn.MaxPool1d(
            kernel_size=1, stride=max_pool_stride))

        self.add_module("gru_encoder", nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=gru_layers,
            bidirectional=True))

        self.add_module("dropoutFCL", nn.Dropout(dropout))
        self.add_module("GRU_FCL", nn.Linear(2*hidden_size, hidden_size))
        self.add_module("encoder_relu", nn.ReLU())

    def forward(self, X):
        modules = self.named_modules()
        d = dict(modules)

        if len(X.size()) > 3:
            X = X.squeeze()

        X = (d["first_cnn"])(X)  # (N, h/2, L)
        X = d["batch_norm"](X)  # (N, C, L)
        X = F.gelu(X)
        X = (d["second_cnn"])(X)
        X = (d["pool"])(X)

        # N, H_in, L
        X = transpose(X, 1, 2)
        # N, L, H_in
        X = transpose(X, 0, 1)
        # L, N, H_in

        X, hidden = d["gru_encoder"](X)
        # (L, N, h_out * D), (Layers * D, N, h_out)

        X = d["dropoutFCL"](X)
        X = d["GRU_FCL"](X)
        # (L, N, h_out)
        # X = d["encoder_relu"](X)
        X = F.gelu(X)

        hidden = hidden[hidden.size(0) // 2:]
        if len(hidden.size()) == 2:
            hidden.unsqueeze(0)

        return X, hidden


class SpeechAttDecVol3(nn.Module):
    def __init__(self, device, hidden_size=256, gru_layers=4, output_size=10, n_classes=27, dropout=0.1):
        super().__init__()

        self.sequence_length = 38  # 40  # 75  # 560x644
        self.device = device

        self.add_module("dropoutFCL1", nn.Dropout(dropout))
        self.add_module("fc_cat_to_h", nn.Linear(
            2 * hidden_size, hidden_size))

        self.add_module("gru_decoder", nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=gru_layers,
            batch_first=False))

        self.add_module("dropoutFCL2", nn.Dropout(dropout))
        self.add_module("fc_to_output", nn.Linear(
            in_features=hidden_size, out_features=n_classes))

        self.add_module("dropoutFCL3", nn.Dropout(dropout))
        self.add_module("output_quantization", nn.Linear(
            in_features=self.sequence_length, out_features=output_size))


    def att(self, enc_out, dec_h):
        """
        enc_out:  (N, L, h)
        dec_h:    (N, 1, h)
        returns attention weights: (N, L)
        """
        # (N, L, h), (N, 1, h)
        scores = torch.sum(dec_h * enc_out, dim=-1)  # (N, L)
        # scores = F.tanh(scores)  # <-1, 1> sigmoid/tanh?
        # scores = torch.exp(scores)
        sums = torch.sum(scores, dim=-1)  # (N)
        scores = scores / \
            sums.unsqueeze(1).repeat(1, enc_out.size(1))  # (N, L)
        scores = F.softmax(scores, dim=-1)
        return scores

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

    def forward(self, gru_encoder_output, hidden):
        """
        gru_encoder_output: (L, N, h)
        hidden: (1, N, h)
        """
        d = dict(self.named_modules())

        decoder_outputs = None
        for seq_iter in range(1, gru_encoder_output.size(0)):


            # ###################################################################################################
            # # GLOBAL ATTENTION
            # #
            # X = self.att(gru_encoder_output.transpose(0, 1),
            #              hidden[-1].unsqueeze(1)).unsqueeze(2).repeat(1, 1, gru_encoder_output.size(-1)) \
            #     * gru_encoder_output.transpose(0, 1)
            # X = torch.sum(X, dim=1)  # (N, h)

            # X = torch.cat((hidden[-1], X), dim=-1).unsqueeze(0)  # (1, N, 2h)
            # #
            # ###################################################################################################

            ###################################################################################################
            # LOCAL ATTENTION
            #
            from_idx, to_idx = self.get_idxs(seq_iter, gru_encoder_output.size(0))
            X = self.att(gru_encoder_output[from_idx:to_idx].transpose(0, 1),
                         hidden[-1].unsqueeze(1)).unsqueeze(2).repeat(1, 1, gru_encoder_output.size(-1)) \
                * gru_encoder_output[from_idx:to_idx].transpose(0, 1)
            X = torch.sum(X, dim=1)  # (N, h)

            X = torch.cat((hidden[-1], X), dim=-1).unsqueeze(0)  # (1, N, 2h)
            #
            ###################################################################################################

            X = d["dropoutFCL1"](X)
            X = d["fc_cat_to_h"](X)
            # X = F.gelu(X)

            X, hidden = d["gru_decoder"](X, hidden)

            # decoder_output, hidden = d["gru_decoder"](
            #     d["fc_cat_to_h"](cat((hidden,
            #                           bmm(self.att(gru_encoder_output.transpose(0, 1),
            #                                        hidden.transpose(0, 1)).unsqueeze(1),
            #                               gru_encoder_output.transpose(0, 1)).transpose(0, 1)), dim=-1)), hidden)

            if decoder_outputs is None:
                decoder_outputs = X
                # (1, N, h_out)
            else:
                decoder_outputs = vstack((decoder_outputs, (X)))
                # (<2;L>, N, h_out)

        # (L, N, h_out)
        decoder_outputs = d["dropoutFCL2"](decoder_outputs)
        decoder_outputs = d["fc_to_output"](decoder_outputs)
        # (L, N, 26)
        decoder_outputs = decoder_outputs.transpose(0, 1)
        # (N, L, 26)

        # decoder_outputs = F.gelu(decoder_outputs)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        decoder_outputs = d["dropoutFCL3"](decoder_outputs)
        decoder_outputs = d["output_quantization"](
            decoder_outputs.transpose(1, 2)).transpose(1, 2)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs