from torch import tensor, nn, ones, onnx, functional as F, transpose
# import numpy

class QueryEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("gruencoder", nn.GRU(input_size = 1, hidden_size = 1, num_layers = 128))
        self.add_module("grudecoder", nn.GRU(input_size = 1, hidden_size = 1, num_layers = 128))
        # batched input -> output(L,N,H)

    def forward(self, X:str):
        modules = self.named_modules()
        d = dict(modules)

        # X - str
        X = X.lower()
        X = [float(ord(i)) for i in X]
        print("str[" + str(len(X)) + "]")
        # repeat the X sequence in a loop so that len(X) = 256
        X = [X[i % len(X)] for i in range(256)]
        X = tensor([X]) # tensor(1, 256)
        print(X.size())

        Y = (d["gruencoder"]).forward(transpose(X, 0, 1))
        tmp = X
        X = Y[0] # Y: tuple(data, metadata)
        print(X.size())
        Y = tmp

        return X # query embeddings