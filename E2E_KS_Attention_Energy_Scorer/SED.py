from torch import tensor, nn, ones, onnx, functional as F, transpose
# import numpy

class SpeechEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("first_cnn", nn.Conv1d(13, 128, 1))
        # "Number of filters = Number of out_channels."
        # "Kernel size of 3 works fine everywhere"
        self.add_module("second_cnn", nn.Conv1d(128, 256, 1))
        self.add_module("pool", nn.MaxPool1d(1, 2))
        self.add_module("gruencoder", nn.GRU(input_size = 256, hidden_size = 256, num_layers = 256))
        self.add_module("encoder_relu", nn.ReLU())
        self.add_module("grudecoder", nn.GRU(input_size = 256, hidden_size = 256, num_layers = 256))
        # batched input -> output(L,N,H)

    def forward(self, X):
        modules = self.named_modules()
        d = dict(modules)

        print(X.size())
        Y = (d["first_cnn"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp
        Y = (d["second_cnn"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp

        Y = (d["pool"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp
        Y = (d["gruencoder"]).forward(transpose(X, 0, 1))
        tmp = X
        X = Y[0] # Y: tuple(data, metadata)
        print(X.size())
        Y = tmp

        Y = (d["encoder_relu"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp
        return X # speech embeddings