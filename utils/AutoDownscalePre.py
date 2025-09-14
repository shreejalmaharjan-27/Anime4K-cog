import torch.nn as nn
import torch.nn.functional as F

class AutoDownscalePre(nn.Module):
    def __init__(self, factor, lower_thresh=1.2, upper_thresh=2.0, upscale_mode="bilinear"):
        super(AutoDownscalePre, self).__init__()
        self.factor = factor // 2
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.upscale_mode = upscale_mode
    def forward(self, x, screen_width=1920, screen_height=1080):
        h, w = x.shape[2:]
        # RPN expression is so weird to understand. Let assume that ChatGPT is right
        # https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_AutoDownscalePre_x2.glsl#L30
        # https://github.com/bloc97/Anime4K/blob/master/glsl/Upscale/Anime4K_AutoDownscalePre_x4.glsl#L30
        h_ratio = h / screen_height / self.factor
        w_ratio = w / screen_width / self.factor
        if (h_ratio > self.lower_thresh) and \
           (w_ratio > self.lower_thresh) and \
           (h_ratio < self.upper_thresh) and \
           (w_ratio < self.upper_thresh):
           return F.interpolate(x, (screen_height // self.factor, screen_width // self.factor), mode=self.upscale_mode)
        return x