from torch import bmm, nn, transpose, cat, exp, sum, vstack, zeros
import torch.nn.functional as F

class SpeechEncVol5(nn.Module):
    def __init__(self, device, hidden_size, n_mfcc=13):
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

        self.add_module("first_cnn", nn.Conv1d(
            in_channels=n_mfcc, out_channels=self.hidden_size // 2, kernel_size=self.conv_kernel_size))
        # "Number of filters = Number of out_channels."
        # "Kernel size of 3 works fine everywhere"
        self.add_module("second_cnn", nn.Conv1d(
            in_channels=self.hidden_size // 2, out_channels=self.hidden_size, kernel_size=self.conv_kernel_size))
        self.add_module("pool", nn.MaxPool1d(
            kernel_size=1, stride=self.max_pool_stride))
        self.add_module("gru_encoder", nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.gru_n_layers, bidirectional=True))
        self.add_module("encoder_relu", nn.ReLU())

    def forward(self, X):
        """
        N - batch_size, L - sequence_length, h - hidden_size (= GRU_input_size)
        input(X): (N, n_MFCC, L) // n_MFCC = 13 by default
        output: (L, N, h)
        """
        modules = self.named_modules()
        d = dict(modules)

        # some MFCC inputs come in shape of (N, 1, 13, L)
        X = X.squeeze()

        X = (d["first_cnn"])(X)

        X = (d["second_cnn"])(X)

        X = (d["pool"])(X)
        # (N, h_in, L) // L = L/2

        unsqueezed = False
        if len(X.size()) == 2:
            X = X.unsqueeze(0)
            unsqueezed = True
        # now X is conceptually batched
        batch_size = X.size(0)

        X = transpose(X, 1, 2)
        # (N, L, h_in)
        X = transpose(X, 0, 1)
        # (L, N, h_in)

        # gru wants: L,N,h_in <- sequence_length, batch, hidden_size
        # hidden: Layers(2), N, h_out // 2 bc bidirectional

        hidden = zeros(2, batch_size, self.hidden_size,
                       device=self.device)
        X, hidden = d["gru_encoder"](X, hidden)
        X = X.view(X.size(0), batch_size, 2, self.hidden_size)[:,:,0]

        X = d["encoder_relu"](X)
        # L, N, h_out

        # we ignore the first hidden state of bidir-GRU
        last_layer = -1
        if unsqueezed:
            return (X[:, 0], hidden[last_layer][0])
        else:
            return (X, hidden[last_layer].unsqueeze(0))


class SpeechAttDecVol5(nn.Module):
    def __init__(self, device, hidden_size, output_size=10):
        """
        hidden_size = 256
        fixed output size of 10 characters
        """
        super().__init__()
        # depends on the encoder behaviour - hard to set dynamically
        self.sequence_length = 38
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_n_layers = 1
        self.device = device

        self.add_module("fc_cat_to_h", nn.Linear(
            2 * self.hidden_size, self.hidden_size))

        self.add_module("gru_decoder", nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.gru_n_layers))

        self.add_module("fc_to_output", nn.Linear(
            in_features=self.hidden_size, out_features=27))

        self.add_module("output_quantization", nn.Linear(
            in_features=self.sequence_length, out_features=self.output_size))  # , bias=False

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
        # ret = F.softmax(exp(bmm(dec_h, enc_out.transpose(1, 2)).squeeze()) /
        #                 sum(exp(bmm(enc_out, dec_h.transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2), dim=-1)
        ret = exp(bmm(dec_h, enc_out.transpose(1, 2)).squeeze(
        )) / sum(exp(bmm(enc_out, dec_h.transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2)
        # print(ret.size()) # (N,L) confirmed
        return ret
        # (N, L)

    def forward(self, gru_encoder_output, hidden):
        """
        gru_encoder_output: (L, N, h)
        hidden: (1, N, h)
        """
        d = dict(self.named_modules())

        decoder_outputs = None
        for _ in range(1, gru_encoder_output.size(0)):
            # cat
            # bmm: (N, 1, L)*(N, L, h) -> (N, 1, h)
            decoder_output, hidden = d["gru_decoder"](
                d["fc_cat_to_h"](cat((hidden,
                                      bmm(self.att(gru_encoder_output.transpose(0, 1), hidden.transpose(0, 1)).unsqueeze(1),
                                          gru_encoder_output.transpose(0, 1)).transpose(0, 1)), dim=-1)), hidden)

            if decoder_outputs is None:
                decoder_outputs = decoder_output
                # (1, N, h_out)
            else:
                decoder_outputs = vstack((decoder_outputs, (decoder_output)))
                # (<2;L>, N, h_out)

            # if decoder_output.equal(EOS):
            #     break

        # (L, N, h_out)
        decoder_outputs = d["fc_to_output"](decoder_outputs)
        # (L, N, 26)
        decoder_outputs = decoder_outputs.transpose(0, 1)
        # (N, L, 26)

        decoder_outputs = d["output_quantization"](
            decoder_outputs.transpose(1, 2)).transpose(1, 2)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs
