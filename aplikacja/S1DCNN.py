import torch
import torch.nn as nn


class S1DCNN(torch.nn.Module):
    def __init__(self):
        super(S1DCNN, self).__init__()
        self.input_dim = 13 # number of mfcc log-mel channels used in pre-processing
        self.layer_num = 35  # number of stacked layers with one for each target class in dataset
        self.k = 9 # size of kernel for time based convolution
        self.n = 32 # number of filters in convolution feature extarction
        self.context = 30 # length of context for deciosion making

        self.feature_layers = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        self.identity = nn.Identity()
        self.relu = nn.ReLU()

        self.linear_layers = nn.ModuleList()

        for n in range(self.layer_num): # declaring layers in a loop so as to stack them
            self.feature_layers.append(nn.Conv1d(in_channels=self.input_dim,
                                                 out_channels=self.n,
                                                 kernel_size=1))
            self.time_layers.append(nn.Conv1d(in_channels=self.n,
                                              out_channels=self.n,
                                              kernel_size=self.k))
            self.linear_layers.append(nn.Linear(in_features=self.context * self.n, out_features=1))

    def forward(self, data):
        x = None
        for i in range(len(self.feature_layers)):
            # 1st layer + activation
            t = self.identity(self.feature_layers[i](data))
            # 2nd layer + activation
            t = self.relu(self.time_layers[i](t))

            # linear classification
            if len(t.size()) > 2:
                t = torch.flatten(t, start_dim=1)[:, -(self.context * self.n):]
            else:
                t = torch.flatten(t, start_dim=0)[-(self.context * self.n):]
            t = self.linear_layers[i](t)
            if x is None:
                x = t
            else:
                x = torch.cat((x, t), dim=-1)

        return x