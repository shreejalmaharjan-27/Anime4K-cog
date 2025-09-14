import torch
import torch.nn as nn
import torch.nn.functional as F

from .MaxPoolKeepShape import MaxPoolKeepShape

def get_luma(x):
    x = x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114
    x = x.unsqueeze(1)
    return x

# Ref: https://github.com/bloc97/Anime4K/blob/master/glsl/Restore/Anime4K_Clamp_Highlights.glsl
class ClampHighlight(nn.Module):
    def __init__(self):
        super(ClampHighlight, self).__init__()
        self.max_pool = MaxPoolKeepShape(kernel_size=(5, 5), stride=1)
    def forward(self, shader_img, orig_img):
        curr_luma = get_luma(shader_img)
        statsmax = self.max_pool(get_luma(orig_img))
        if statsmax.shape != curr_luma.shape:
            statsmax = F.interpolate(statsmax, curr_luma.shape[2:4])
        new_luma = torch.min(curr_luma, statsmax)
        return shader_img - (curr_luma - new_luma)