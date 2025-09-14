import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

from utils.CReLU import CReLU

def conv_layer(in_channels, out_channels, kernel_size):
    padding = int((kernel_size - 1) / 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

def convert(c, iter, doswap=False):
    swap = [0,2,1,3]
    out_chan, in_chan, width, height = c.weight.shape
    for to in range(math.ceil(out_chan/4)):
        for ti in range(math.ceil(in_chan/4)):
            for w in range(width):
                for h in range(height):
                    for i in range(min(4, in_chan)):
                        for o in range(min(4, out_chan)):
                            o = swap[o] if doswap else o
                            c.weight.data[to*4+o, ti*4+i, w, h] = float(next(iter).group(0))
        for o in range(min(4, out_chan)):
            o = swap[o] if doswap else o
            c.bias.data[to*4+o] = float(next(iter).group(0))

# Credit: https://github.com/kato-megumi
# block_depth: Num of hidden layer shaders + 1, must be >= num_feat
# num_feat: Total output channel of input Conv2D shaders
# stack_list = Num of binds of last_conv2d / 2
# last: last_conv2d's kernel=3 if True else 1
# single_tail: Set this to True if there is only 1 conv2d_last shader (e.g. Upscale_S, Upscale_M)
# Input conv2d shaders are first shaders having "MAIN_texOff", output conv2d shaders are last ones having "conv2d_last_tf"
class anime4k(nn.Module):
    def __init__(self, block_depth=7, stack_list=5, num_feat=12, last=False, scale=2, single_tail=False, upscale_mode="bilinear"):
        super(anime4k, self).__init__()
        self.act = CReLU()
        if isinstance(stack_list, int):
            self.stack_list = list(range(-stack_list, 0))
        else:
            self.stack_list = stack_list
        self.scale = scale
        self.ps = nn.PixelShuffle(self.scale)
        self.conv_head = conv_layer(3, num_feat, kernel_size=3)
        self.conv_mid = nn.ModuleList(
            [
                conv_layer(num_feat * 2, num_feat, kernel_size=3)
                for _ in range(block_depth - 1)
            ]
        )
        tail_out_c = 4 if single_tail else 3*scale*scale
        if last:
            self.conv_tail = conv_layer(2 * num_feat * len(self.stack_list), tail_out_c, kernel_size=3)
        else:
            self.conv_tail = conv_layer(2 * num_feat * len(self.stack_list), tail_out_c, kernel_size=1)
        self.upscale_mode = upscale_mode


    def forward(self, x):
        out = self.act(self.conv_head(x))
        depth_list = [out]
        for conv in self.conv_mid:
            out = self.act(conv(out))
            depth_list.append(out)
        out = self.conv_tail(torch.cat([depth_list[i] for i in self.stack_list], 1))
        if self.scale != 1:
            out = self.ps(out) + F.interpolate(x, scale_factor=self.scale, mode=self.upscale_mode)
        else:
            out += x
        return torch.clamp(out, max=1.0, min=0.0)

    def import_param(self, filename):
        for param in self.parameters():
            param.requires_grad = False
        with open(filename) as f:
            text = f.read()
        pattern = r'-?\d+(\.\d{4,})(e-?\d+)?'
        iter = re.finditer(pattern, text)
        convert(self.conv_head, iter)
        for conv in self.conv_mid:
            convert(conv, iter)
        convert(self.conv_tail, iter, True)
        check = next(iter, None)
        if check == None:
            print("pass")
        else:
            print("---failed---\n", check)
