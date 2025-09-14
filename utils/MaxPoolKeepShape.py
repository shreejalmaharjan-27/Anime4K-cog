import torch.nn as nn
import torch.nn.functional as F

class MaxPoolKeepShape(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPoolKeepShape, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        kernel_height, kernel_width = self.kernel_size
        pad_height = (((height - 1) // self.stride + 1) - 1) * self.stride + kernel_height - height
        pad_width = (((width - 1) // self.stride + 1) - 1) * self.stride + kernel_width - width

        x = F.pad(x, (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2))
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x