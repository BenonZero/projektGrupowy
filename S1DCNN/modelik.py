import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Stacked1DCNN(nn.Module):
    def __init__(self,number_of_vectors_layer_1,number_of_vectors_layer_2,number_of_filters,kernel_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=number_of_vectors_layer_1, out_channels=number_of_filters, kernel_size=kernel_size),
            # layer1
            nn.ReLU(),
            nn.Conv1d(in_channels=number_of_vectors_layer_2, out_channels=number_of_filters, kernel_size=kernel_size)
            # layer2
        )
        self.fc=nn.LazyLinear(1)
        self.softmax=nn.Softmax(dim=0)
        self.flatten=nn.Flatten(start_dim=1)
    def forward(self, x):
        networks = []
        for i in range(7):
            networks.append(self.model(x))
        stacked=torch.stack(networks,1)
        stacked=self.flatten(stacked)
        stacked=self.fc(stacked)
        return F.log_softmax(stacked, dim=0)

