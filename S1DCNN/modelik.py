import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Stacked1DCNN(nn.Module):
    def __init__(self,number_of_vectors_layer_1,number_of_vectors_layer_2,number_of_filters,kernel_size):
        super().__init__()
        self.layer1=nn.Conv1d(in_channels=number_of_vectors_layer_1, out_channels=number_of_filters, kernel_size=kernel_size)     #layer1
        self.relu=nn.ReLU()
        self.layer2=nn.Conv1d(in_channels=number_of_vectors_layer_2, out_channels=number_of_filters, kernel_size=kernel_size)     #layer2
        self.fc=nn.LazyLinear(1)
        self.softmax=nn.Softmax(dim=0)
        self.flatten=nn.Flatten(start_dim=1)
    def forward(self, x):
        #networks = []
        #for i in range(7):
            #networks.append(self.model(x))
        x1=self.layer1(x)
        x2=self.layer1(x)
        x3=self.layer1(x)
        x4=self.layer1(x)
        x5=self.layer1(x)
        x6=self.layer1(x)
        x7=self.layer1(x)
        x1=self.relu(x1)
        x2=self.relu(x2)
        x3=self.relu(x3)
        x4=self.relu(x4)
        x5=self.relu(x5)
        x6=self.relu(x6)
        x7=self.relu(x7)
        x1 = self.layer2(x1)
        x2 = self.layer2(x2)
        x3 = self.layer2(x3)
        x4 = self.layer2(x4)
        x5 = self.layer2(x5)
        x6 = self.layer2(x6)
        x7 = self.layer2(x7)
        stacked=torch.stack((x1,x2,x3,x4,x5,x6,x7),1)
        stacked=self.flatten(stacked)
        stacked=self.fc(stacked)
        return F.log_softmax(x1, dim=0)

