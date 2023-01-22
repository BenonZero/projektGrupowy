from python_speech_features import mfcc
from python_speech_features import logfbank
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
        # self.add_module() RELU
        self.add_module("grudecoder", nn.GRU(input_size = 256, hidden_size = 256, num_layers = 256))
        # batched input -> output(L,N,H)

    def forward(self, X):
        modules = self.named_modules()
        d = dict(modules)
        # print(d)
        # for idx, module in enumerate(modules):
        #     print(module[0])
        # print(modules)
        # Y = []
        # for x in X:
        #     tmp = [[x]]
        #     Y.append((d["first_cnn"]).forward(tensor(tmp)))

        # print(d["first_cnn"])
        # print(tensor(X))
        # print(ones(1, 13, 1))
        # print(type(ones(1)[0]))
        # print(tensor([X[0]]))
        # print((d["first_cnn"]).forward(ones(1, 13, 1))) #.forward(tensor([X])))
        # print((d["first_cnn"]).forward(tensor([X[0]])))
        print(X.size())
        # print(len(X[0][0]))
        # Y = [(d["first_cnn"]).forward(tensor([x])) for x in X]
        Y = (d["first_cnn"]).forward(X)
        # print(Y)
        tmp = X
        X = Y
        print(X.size())
        # print(len(X[0][0]))
        Y = tmp
        # print(X)
        # X = [[[X[j][0][i]] for i in range(len(X[0][0]))] for j in range(len(X))]
        # print(len(X))
        # print(len(X[0]))
        # print(len(X[0][0]))
        # Y = [(d["second_cnn"]).forward(tensor([x])) for x in X]
        Y = (d["second_cnn"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        # print(len(X[0][0]))
        Y = tmp
        # print(X)
        # X = [[[X[j][0][i]] for i in range(len(X[0][0]))] for j in range(len(X))]
        # print(len(X))
        # print(len(X[0]))
        # print(len(X[0][0]))
        # print(len(Y))
        # print(len(Y[0]))
        # print(len(Y[0][0]))
        # print(Y)
        # print(X)

        Y = (d["pool"]).forward(X)
        tmp = X
        X = Y
        print(X.size())
        # print(len(X[0][0]))
        Y = tmp
        # print(X)
        # X = [[[X[j][0][i]] for i in range(len(X[0][0]))] for j in range(len(X))]
        # print(len(X))
        # print(len(X[0]))
        # print(len(X[0][0]))

        Y = (d["gruencoder"]).forward(transpose(X, 0, 1))
        # input.size(-1) must be equal to input_size
        tmp = X
        X = Y
        print(X[0].size())
        Y = tmp

        return X[0]


sed = SpeechEncDec()
onnx.export(sed, \
    tensor([[float(i+j) for i in range(5)] for j in range(13)]),\
        "module.onnx")
# sed.forward([[[float(i+j)] for i in range(13)] for j in range(5)])
sed.forward(tensor([[float(i+j) for i in range(5)] for j in range(13)]))


# from python_speech_features import mfcc
# from python_speech_features import logfbank
# import scipy.io.wavfile as wav
# (rate,sig) = wav.read(".wav")
# mfcc_feat = mfcc(sig,rate) # -> array[n][13]
# fbank_feat = logfbank(sig, rate)