from torch import cat, nn

class AttMech(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("LSTM", nn.LSTM(input_size = 512, hidden_size = 256, bidirectional = True)) #, num_layers = 128 TODO: spojnosc z grafika, doczytanie
        self.add_module("FCL", nn.Linear(in_features = 512, out_features = 1))
        self.add_module("sigmoid", nn.Sigmoid())

    def forward(self, E_s, E_q):
        modules = self.named_modules()
        d = dict(modules)
        # e_q: (1,256), e_s: (x, 256)
        catted = cat((E_s, E_q.repeat(E_s.size()[0], 1)), 1)
        print("concat(" + str(E_s.size()) + ", " + str(E_s.size()[0]) + " * " + str(E_q.size()))
        print(catted.size())

        catted = (d["LSTM"]).forward(catted)[0] # tuple(data, metadata)
        print(catted.size())

        catted = (d["FCL"]).forward(catted)
        print(catted.size())

        catted = (d["sigmoid"]).forward(catted)
        print(catted.size())

        return catted # attention weights