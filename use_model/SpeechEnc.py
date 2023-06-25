from torch import nn, transpose
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
