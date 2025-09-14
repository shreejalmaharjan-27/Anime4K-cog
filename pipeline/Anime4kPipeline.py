import torch
import torch.nn as nn
import torch.nn.functional as F

class Anime4KPipeline(nn.Module):
    def __init__(self, *models, final_stage_upscale_mode="bilinear"):
        super(Anime4KPipeline, self).__init__()
        self.models = nn.ModuleList(models)
        self.final_stage_upscale_mode = final_stage_upscale_mode

    def half(self):
        return self.to(torch.half)
    def forward(self, x, screen_width=1920, screen_height=1080):
        clamp_hightlight, orig_img = None, None
        for model in self.models:
            if model.__class__.__name__ == "AutoDownscalePre":
                x = model(x, screen_width, screen_height)
                continue
            if model.__class__.__name__ == "ClampHighlight":
                clamp_hightlight = model
                orig_img = x.clone()
                continue
            x = model(x)
        if clamp_hightlight is not None:
            x = clamp_hightlight(x, orig_img)
        x = F.interpolate(x, (screen_height, screen_width), mode=self.final_stage_upscale_mode)
        return x