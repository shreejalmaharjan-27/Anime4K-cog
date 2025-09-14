import torch
import torch.nn as nn
import torch.nn.functional as F

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return F.relu(torch.cat((x, -x), 1))