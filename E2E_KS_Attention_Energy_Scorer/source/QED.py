from torch import tensor, nn, transpose
# import numpy

class QueryEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("gru_encoder", nn.GRU(input_size = 1, hidden_size = 1, num_layers = 128))
        self.add_module("gru_decoder", nn.GRU(input_size = 1, hidden_size = 1, num_layers = 128))
        # batched input -> output(L,N,H)

    def forward(self, X:str):
        modules = self.named_modules()
        d = dict(modules)

        # X - str
        X = X.lower()
        X = [float(ord(i)) for i in X]
        print("str[" + str(len(X)) + "]")
        # repeat the X sequence in a loop so that len(X) = 256
        # alternatively spread the X sequence to the desired length
        X = [X[i % len(X)] for i in range(256)]
        X = tensor([X]) # tensor(1, 256)
        print(X.size())

        Y = (d["gru_encoder"]).forward(transpose(X, 0, 1))
        tmp = X
        X = transpose(Y[0], 0, 1) # Y: tuple(data, metadata)
        print(X.size())
        Y = tmp

        return X # query embeddings
    
        #TODO - DECODER
        # X -> gru_ENCODER -> gru_DECODER -> X (taki sam)