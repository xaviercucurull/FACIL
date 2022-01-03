import torch.nn as nn
import torch.nn.functional as F

class XaviNet(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.fc1 = nn.Linear(420, 64)
        self.fc2 = nn.Linear(64, 32)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=32, out_features=num_classes)
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


def xavinet(num_out=11, pretrained=False):
    if pretrained:
        raise NotImplementedError
    return XaviNet(num_out)
