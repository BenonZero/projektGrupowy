from torch import Tensor, bmm, nn, ones, stack, tanh, transpose, dot, tensor, softmax, cat, tensor, exp, sum, empty, vstack, zeros
import torch.nn.functional as F


# TODO:
# > zmniejszac wyjscie warstwa FCL albo Conv1D zamiast poszerzac target w trainingu
# > def test()
# > testowanie reczne: [batch, seq, probabilities] -> [batch, seq, most_probable_char] -> batch of char sequences to print out


class SpeechEncVol3(nn.Module):
    def __init__(self, device, n_mfcc=13, input_size=256, enc_h_size=256):
        super().__init__()

        self.input_size = input_size
        self.encoder_hidden_size = enc_h_size
        self.device = device
        self.max_pool_stride = 2
        self.gru_n_layers = 1

        self.add_module("first_cnn", nn.Conv1d(
            in_channels=n_mfcc, out_channels=self.input_size // 2, kernel_size=1))
        # "Number of filters = Number of out_channels."
        # "Kernel size of 3 works fine everywhere"
        self.add_module("second_cnn", nn.Conv1d(
            in_channels=self.input_size // 2, out_channels=self.input_size, kernel_size=1))
        self.add_module("pool", nn.MaxPool1d(
            kernel_size=1, stride=self.max_pool_stride))
        # TODO: more layers in GRUs?
        self.add_module("gru_encoder", nn.GRU(
            input_size=self.input_size, hidden_size=self.encoder_hidden_size, num_layers=self.gru_n_layers, batch_first=False))
        self.add_module("encoder_relu", nn.ReLU())

    def forward(self, X):
        modules = self.named_modules()
        d = dict(modules)

        X = X.squeeze()
        # print(f"\ninput shape: {tuple(X.size())}")
        X = (d["first_cnn"])(X)
        # print(f"after first_cnn: {tuple(X.size())}")
        X = (d["second_cnn"])(X)
        # print(f"after second_cnn: {tuple(X.size())}")
        X = (d["pool"])(X)
        # print(f"after pooling: {tuple(X.size())}")
        # torch.Size([2, 256, 128]) <- batch, input_size, seq_length

        unsqueezed = False
        if len(X.size()) == 2:
            X = X.unsqueeze(0)
            unsqueezed = True
        # now X is conceptually batched
        batch_size = X.size()[0]

        # N, H_in, L
        X = transpose(X, 1, 2)
        # N, L, H_in
        X = transpose(X, 0, 1)
        # L, N, H_in

        # gru wants: L,N,H_in <- sequence_length, batch, input_size
        # h_in: Layers, N, h_out

        hidden = zeros(1, batch_size, self.encoder_hidden_size,
                       device=self.device)
        X, hidden = d["gru_encoder"](
            X, hidden)

        X = d["encoder_relu"](X)
        # L, N, H_out

        last_layer = -1
        if unsqueezed:
            return (X[:, 0], hidden[last_layer][0])
        else:
            return (X, hidden)


class SpeechAttDecVol3(nn.Module):
    def __init__(self, device, dec_h_size=256, seq_len=128, input_size=256, output_size=10) -> None:
        super().__init__()
        # self.add_module("first_cnn", nn.Conv1d(in_channels=13, out_channels=128, kernel_size=1))
        # self.sequence_length = seq_len
        self.sequence_length = 40
        self.decoder_hidden_size = dec_h_size
        self.input_size = input_size
        self.output_size = output_size
        self.max_pool_stride = 2
        self.gru_n_layers = 1

        self.add_module("fc_cat_to_h", nn.Linear(
            2 * self.decoder_hidden_size, self.decoder_hidden_size))

        self.add_module("gru_decoder", nn.GRU(
            input_size=self.decoder_hidden_size, hidden_size=self.decoder_hidden_size, num_layers=self.gru_n_layers, batch_first=False))

        self.add_module("fc_to_output", nn.Linear(
            in_features=self.decoder_hidden_size, out_features=27))
        
        self.add_module("output_quantization", nn.Linear(
            in_features=self.sequence_length, out_features=self.output_size)) #, bias=False
        
        # self.add_module("fc_to_output", nn.Linear(
        #     in_features=self.decoder_hidden_size, out_features=26))

        # self.add_module("output_quantization", nn.Linear(
            # in_features=self.sequence_length, out_features=self.output_size, bias=False)) #

        self.device = device

    def att(self, enc_out, dec_h):
        """
        enc_out:  (N, L, h)
        dec_h:    (N, 1, h)
        returns attention weights: (N, L)
        """
        # TODO: softmax na koncu?
        # bmm: (N,1,h)*(N,h,L)-> (N, 1, L) -> (N,L)
        # (N,L)/()
        ret = F.softmax(exp(bmm(dec_h, enc_out.transpose(1, 2)).squeeze()) /
                        sum(exp(bmm(enc_out, dec_h.transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2), dim=-1)
        # print(ret.size()) # (N,L) confirmed
        return ret
        # bmm (N,L,h)*(N,h,L) -> (N,L,L) -> (N,L)
        # (N, L)

    def forward(self, gru_encoder_output, hidden):
        """
        gru_encoder_output: (L, N, h)
        hidden: (1, N, h)
        """
        d = dict(self.named_modules())
        
        decoder_outputs = None
        for seq_iter in range(1, gru_encoder_output.size(0)):



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

        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        decoder_outputs = d["output_quantization"](decoder_outputs.transpose(1, 2)).transpose(1, 2)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs
