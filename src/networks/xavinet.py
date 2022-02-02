import torch.nn as nn
import torch.nn.functional as F

class XaviNet(nn.Module):
    """ Simple Neural Network."""

    def __init__(self, in_size=420, num_classes=11, **kwargs):
        super().__init__()
        # main part of the network
        self.fc1 = nn.Linear(in_size, 64)
        self.fc2 = nn.Linear(64, 32)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(32, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x
