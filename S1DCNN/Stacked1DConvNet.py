import torch
import torch.nn as nn

class Stacked1DConvNet(nn.Module):
    def __init__(self, K, L):
        super(Stacked1DConvNet, self).__init__()
        self.identity_convs = nn.ModuleList([
          nn.Conv1d(in_channels=13, 
                    out_channels=32, 
                    kernel_size=K+L+1,  # Wykorzystywane jest K-L-1 poprzednich wyjść, aktualne oraz L następnych wyjść.
                                        # Wartość do przetestowania
                    padding=(K+L)//2, 
                    groups=13)  # Wejściem dla warstwy drugiej są wyjścia warstwy pierwszej
                                # Wartość do przetestowania
          for _ in range(7)])
        self.relu_convs = nn.ModuleList([
          nn.Conv1d(in_channels=32, 
                    out_channels=32, 
                    kernel_size=K+L+1,  # Wykorzystywane jest K-L-1 poprzednich wyjść, aktualne oraz L następnych wyjść.
                                        # Wartość do przetestowania
                    padding=(K+L)//2, 
                    groups=32)  # Wejściem dla warstwy drugiej są wyjścia warstwy pierwszej
                                # Wartość do przetestowania
          for _ in range(7)])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 20)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for conv in self.identity_convs:
            x = conv(x)
        for conv in self.relu_convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
