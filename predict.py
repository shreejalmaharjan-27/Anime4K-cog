import uuid
from cog import BasePredictor, Input, Path
from PIL import Image
import locale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from pipeline.Anime4kPipeline import Anime4KPipeline
from utils.AutoDownscalePre import AutoDownscalePre
from utils.ClampHighlight import ClampHighlight
from utils.anime4k import anime4k

from einops import repeat

from utils.util import timed

locale.getpreferredencoding = lambda: "UTF-8"

if not os.path.exists("/tmp/libtcmalloc_minimal.so.4"):
    os.system("wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /tmp/libtcmalloc_minimal.so.4 -nc")

os.environ['LD_PRELOAD'] = "/tmp/libtcmalloc_minimal.so.4:" + os.environ.get('LD_PRELOAD', '')
os.environ['LC_ALL'] = "en_US.UTF-8"

paths_to_add = []
try:
    import tensorrt
    tensorrt_path = os.path.dirname(tensorrt.__file__)
    tensorrt_libs_path = os.path.join(os.path.dirname(tensorrt_path), 'tensorrt_libs')
    if os.path.exists(tensorrt_libs_path):
        paths_to_add.append(tensorrt_libs_path)
except ImportError:
    pass

new_ld_library_path = ":".join(paths_to_add)
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

os.environ['LD_LIBRARY_PATH'] = f"{new_ld_library_path}:{ld_library_path}" if ld_library_path else new_ld_library_path
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])
os.environ['LIBRARY_PATH'] = "/usr/local/cuda/lib64/stubs"
os.system(f"ldconfig {new_ld_library_path}")

class Predictor(BasePredictor):
    def setup(self, weights = None) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.to_pil = torchvision.transforms.ToPILImage()
        self.to_tensor = torchvision.transforms.ToTensor()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_dict = {
            "Upscale_S": ("glsl/Anime4K_Upscale_CNN_x2_S.glsl", 3, 1, 4, True, 2, True),
            "Upscale_M": ("glsl/Anime4K_Upscale_CNN_x2_M.glsl", 7, 7, 4, False, 2, True),
            "Upscale_L": ("glsl/Anime4K_Upscale_CNN_x2_L.glsl", 3, 1, 8, True, 2),
            "Upscale_VL": ("glsl/Anime4K_Upscale_CNN_x2_VL.glsl", 7, 7, 8, False, 2),
            "Upscale_UL": ("glsl/Anime4K_Upscale_CNN_x2_UL.glsl", 7, 5, 12, False, 2),
            "Upscale_Denoise_S": ("glsl/Anime4K_Upscale_Denoise_CNN_x2_S.glsl", 3, 1, 4, True, 2, True),
            "Upscale_Denoise_M": ("glsl/Anime4K_Upscale_Denoise_CNN_x2_M.glsl", 7, 7, 4, False, 2, True),
            "Upscale_Denoise_L": ("glsl/Anime4K_Upscale_Denoise_CNN_x2_L.glsl", 3, 1, 8, True, 2),
            "Upscale_Denoise_VL": ("glsl/Anime4K_Upscale_Denoise_CNN_x2_VL.glsl", 7, 7, 8, False, 2),
            "Upscale_Denoise_UL": ("glsl/Anime4K_Upscale_Denoise_CNN_x2_UL.glsl", 7, 5, 12, False, 2),
            "Restore_S": ("glsl/Anime4K_Restore_CNN_S.glsl", 3, 1, 4, True, 1),
            #"Restore_M": ("glsl/Anime4K_Restore_CNN_M.glsl", 7, 7, 4, False, 1), Doesn't work for some reason
            "Restore_L": ("glsl/Anime4K_Restore_CNN_L.glsl", 4, 1, 8, True, 1),
            "Restore_VL": ("glsl/Anime4K_Restore_CNN_VL.glsl", 8, 7, 8, False, 1),
            "Restore_UL": ("glsl/Anime4K_Restore_CNN_UL.glsl", 8, 5, 12, False, 1),
            "Restore_Soft_S": ("glsl/Anime4K_Restore_CNN_Soft_S.glsl", 3, 1, 4, True, 1),
            "Restore_Soft_M": ("glsl/Anime4K_Restore_CNN_Soft_M.glsl", 7, 7, 4, False, 1),
            "Restore_Soft_L": ("glsl/Anime4K_Restore_CNN_Soft_L.glsl", 4, 1, 8, True, 1),
            "Restore_Soft_VL": ("glsl/Anime4K_Restore_CNN_Soft_VL.glsl", 8, 7, 8, False, 1),
            "Restore_Soft_UL": ("glsl/Anime4K_Restore_CNN_Soft_UL.glsl", 8, 5, 12, False, 1)
        }


        self.pipelines = {
            "A (Fast)": Anime4KPipeline(
                ClampHighlight(),
                self.create_model("Upscale_Denoise_M"),
                AutoDownscalePre(4),
                self.create_model("Upscale_S"),
                final_stage_upscale_mode = "bilinear"
            ),
            "B (Balanced)": Anime4KPipeline(
                ClampHighlight(),
                self.create_model("Upscale_Denoise_L"),
                AutoDownscalePre(4),
                self.create_model("Upscale_S"),
                final_stage_upscale_mode = "bilinear"
            ),
            "C (Quality)": Anime4KPipeline(
                ClampHighlight(),
                self.create_model("Upscale_Denoise_VL"),
                AutoDownscalePre(4),
                self.create_model("Upscale_M"),
                final_stage_upscale_mode = "bilinear"
            )
        }

        for name in self.pipelines.keys():
            self.pipelines[name] = self.pipelines[name].to(self.device).half()


        # pre flight checks
        rand_image = torch.rand(1, 3, 720, 1280).to(self.device).half()

        # Use one of the pipelines for the check
        test_pipeline = self.pipelines["C (Quality)"]
        test_pipeline(rand_image)
        for _ in range(5):
            _, t = timed(lambda: test_pipeline(rand_image))
            print(t * 1000, 'ms')

        onnx_filename = "onnx/pipeline_preset_C.onnx" #@param {"type": "string"}
        if not os.path.exists("onnx"):
            os.makedirs("onnx")

        torch.onnx.export(
            test_pipeline,
            (rand_image,),
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    def create_model(self, name, upscale_mode="bilinear"):
        filename, *model_params = self.model_dict[name]
        model = anime4k(*model_params, upscale_mode=upscale_mode)
        model.import_param(filename)
        return model
    
    
    def predict(
        self,
        image: Path = Input(description="Image to Process"),
        preset: str = Input(
            description="Model preset to use",
            choices=["A (Fast)", "B (Balanced)", "C (Quality)"],
            default="C (Quality)",
        ),
        resolution: str = Input(
            description="Output resolution",
            choices=["1080p", "2K", "4K", "8K"],
            default="1080p",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        resolution_map = {
            "1080p": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160),
            "8K": (7680, 4320),
        }
        width, height = resolution_map[resolution]

        img = Image.open(str(image)).convert("RGB")
        image_tensor = self.to_tensor(img).unsqueeze(0).to(self.device).half()

        id = str(uuid.uuid4())[:8]

        orig_w, orig_h = img.size

        # Preserve aspect ratio
        ratio = min(width / orig_w, height / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        pipeline = self.pipelines[preset]

        out = pipeline(image_tensor, screen_width=new_w, screen_height=new_h)
        out = self.to_pil(out.squeeze().cpu())
        out.save(f"/tmp/out_{id}.png")

        return Path(f"/tmp/out_{id}.png")