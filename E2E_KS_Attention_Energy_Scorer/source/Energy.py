from torch import nn, dot, sum

class EneSc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # MLP with two layers as in:
        # https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
        self.add_module("MLP_FCL_IN", nn.Linear(in_features = 256, out_features = 128))
        # self.add_module("MLP_FCL_H", nn.Linear(in_features = 128, out_features = 1))
        self.add_module("MLP_FCL_OUT", nn.Linear(in_features = 128, out_features = 1))
        self.add_module("sigmoid", nn.Sigmoid())
        self.add_module("ReLU", nn.ReLU())

    def forward(self, E_s, E_q, Att_weights):
        modules = self.named_modules()
        d = dict(modules)
        # e_q: (1,256), e_s: (x, 256)

        energy_s = 0
        for t in range(E_s.size()[0]):
            energy_s += dot(E_s[t], E_s[t])
        for t1 in range(E_s.size()[0]):
            for t2 in range(E_s.size()[0]):
                if (t1 != t2):
                    energy_s += dot(E_s[t1], E_s[t2])

        context = E_s
        for t in range(E_s.size()[0]):
            E_s[t] *= Att_weights[t]
        context = sum(context, 0)
        energy_c = dot(context, context)

        r = energy_c / energy_s
        # energy_c <= energy_s
        # therefore: use sigmoid for the treshold
        # One hidden layer is sufficient for a large majority of problems

        # https://randlow.github.io/posts/machine-learning/num-layers-neurons/:
        # The optimal size of the hidden layer (i.e., number of neurons)
        # is between the size of the input and the size of the output layer.
        # A good start is to use the average of the total number of neurons
        # in both the input and output layers

        print(E_q.size())
        E_q = (d["MLP_FCL_IN"]).forward(E_q)
        print(E_q.size())
        E_q = (d["ReLU"]).forward(E_q)
        print(E_q.size())
        E_q = (d["MLP_FCL_OUT"]).forward(E_q)
        print(E_q.size())
        E_q = (d["sigmoid"]).forward(E_q)
        print(E_q.size())

        r_th = E_q.item()

        if (r >= r_th):
            return True
        else:
            return False