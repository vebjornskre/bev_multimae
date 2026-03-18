# stdlib
import contextlib
import glob
import logging
import os
import warnings
from dataclasses import replace
from pathlib import Path

# third-party
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image
from transformers import pipeline

# hydra
import hydra
from hydra.utils import to_absolute_path

# local / project
from bev_multimae.visualization.depth_visualization import plot_depth_maps

log = logging.getLogger(__name__)


def load_single_img(cfg: DictConfig):

    img_folder = Path(to_absolute_path(cfg.imgs_raw_path))
    idx = cfg.img_frame

    ext = "*.jpg"
    img_files = glob.glob(os.path.join(img_folder, ext))

    img_path = sorted(img_files)[idx]
    log.info(f"Processing image: {Path(img_path).name}")

    if cfg.depth_model == 'depth_pro':
        import depth_pro
        img, _, _ = depth_pro.load_rgb(img_path)
        return img
    
    img = Image.open(img_path).convert("RGB")
    return img

def cnn_feature_extract(model, img, device):

    model.eval()
    model.to(device)
    backbone = model.features[:16]

    # Preprocessing
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isinstance(img, Image.Image):
        transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        x = transform(img).unsqueeze(0).to(device)

    elif isinstance(img, torch.Tensor):
        if img.dim() == 3:
            x = normalize(img).unsqueeze(0).to(device)
        elif img.dim() == 4:
            x = normalize(img).to(device)
    
    # Extract feature map
    with torch.no_grad():
        feat = backbone(x)

    log.info(f"CNN feature map shape: {feat.shape}")
    return feat

class DepthEstimator:
    def __init__(self, cfg: DictConfig, device: str = None, plot=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.depth_model = cfg.depth_model
        self.cam_info_path = cfg.camera_info
        self.plot_folder = cfg.plot_folder

        if self.depth_model == 'depth_pro':
            self.depth_pro_weights = cfg.depth_pro_weights

        cam_info = np.load(cfg.camera_info)
        self.K = cam_info['K']
        self.plot = plot

    def _load_model(self):
        if self.depth_model == "zoe":
            self.model = self._load_zoe()
            self.transform = None

        elif self.depth_model == "depth_pro":
            self.model, self.transform = self._load_depth_pro()

        elif self.depth_model == "depth_any":
            self.model = self._load_depth_any()
            self.transform = None

        elif self.depth_model == "metric3d":
            self.model = self._load_metric3d()
            self.transform = None
        
        elif self.depth_model == "unidepth":
            self.model = self._load_unidepth()
            self.transform = None

        else:
            raise ValueError(f"Unknown depth_model: {self.depth_model}")

    def _predict(self, img):
        assert hasattr(self, "model") and self.model is not None, "Call _load_model() first (model missing)"
        if self.depth_model == 'zoe': return self._predict_zoe(img)
        elif self.depth_model == 'depth_pro': return self._predict_depth_pro(img)
        elif self.depth_model == 'depth_any': return self._predict_depth_any(img)
        elif self.depth_model == 'metric3d': return self._predict_metric3d(img)
        elif self.depth_model == 'unidepth': return self._predict_unidepth(img)
 
    def _load_zoe(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        _orig_load = nn.Module.load_state_dict
        nn.Module.load_state_dict = lambda self, state_dict, **kw: _orig_load(self, state_dict, strict=False)

        # Load ZoeDepth
        with warnings.catch_warnings(), open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
            warnings.simplefilter("ignore")
            logging.getLogger("torch.hub").setLevel(logging.ERROR)
            zoe = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_NK",
                pretrained=True,
                verbose=False,
            ).to(device)

        # Restore original load_state_dict
        nn.Module.load_state_dict = _orig_load
        log.info("ZoeDepth model loaded")

        return zoe
    
    def _load_depth_pro(self):
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = replace(
            DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=self.depth_pro_weights
        )

        # Load model and preprocessing transform (suppress verbose model print)
        vit_logger = logging.getLogger("depth_pro.network.vit_factory")
        prev_level = vit_logger.level
        vit_logger.setLevel(logging.WARNING)
        model, transform = depth_pro.create_model_and_transforms(config=config)

        log.info("Depth Pro model loaded")

        vit_logger.setLevel(prev_level)
        model.to(device)
        model.eval()

        return model, transform
    
    def _load_depth_any(self):
        return pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
    def _load_metric3d(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Patch cached hub code to avoid mmcv dependency
        self._patch_metric3d_hub()

        # Load model 
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model.to(device).eval()
        log.info("Metric3D v2 ViT-Large model loaded")

        return model
    
    def _load_unidepth(self):
        from unidepth.models import UniDepthV2

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        model.resolution_level = 2

        model.to(device)
        model.eval()

        log.info("UniDepthV2 model loaded")
        log.info(f"Model backbone param abs mean: {next(model.parameters()).abs().mean()}")
        log.info(f"Lenght of model state dict: {len(model.state_dict())}")

        return model

    def _predict_zoe(self, img):
        assert isinstance(img, Image.Image), f"Expected PIL image, got {type(img)}"

        self.model.to(self.device)
        self.model.eval()

        try:
            depth = self.model.infer_pil(img, output_type="tensor")
        except AttributeError:
            raise RuntimeError(
                "\033[93mZoeDepth needs timm version 0.6.12.\n"
                "Activate separate Zoe venv and run again.\n"
                "Commands:\n\033[0m"
                "source .venv_zoe/bin/activate\n"
                "python3 src/bev_multimae/preprocessing/camera/depth.py"
            )
        if self.plot:
            plot_depth_maps(self.cfg, img, depth)

        return depth
    
    def _predict_depth_pro(self, img):
        assert isinstance(img, np.ndarray), f'Expected np.ndarray, got {type(img)}'

        f_px = self.K[0, 0]
        log.info(f"Using calibrated focal length: f_px={f_px:.2f}")

        image = self.transform(img)
        image = image.to(self.device)

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)

        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        if self.plot:
            plot_depth_maps(self.cfg, img, depth, None)
        
        return depth
    
    def _predict_depth_any(self, img):
        assert isinstance(img, Image.Image), f"Expected PIL image, got {type(img)}"

        # Model is actually a pipeline here
        result = self.model(img)  
        depth = result["depth"]  # PIL Image of depth
        depth_tensor = result["predicted_depth"]  # torch.Tensor [1, H, W] in meters

        if self.plot:
            plot_depth_maps(self.cfg, img, depth_tensor, None)

        return depth_tensor

    def _predict_metric3d(self, img):

        ## Data transformations
        intrinsic = [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]]  # [fx, fy, cx, cy]

        rgb_origin = np.array(img)
        h, w = rgb_origin.shape[:2]

        # Resize keeping ratio to fit ViT input size
        input_size = (616, 1064)
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv.INTER_LINEAR)

        # Scale intrinsics to match resized image
        intrinsic_scaled = [v * scale for v in intrinsic]

        # Pad to exact input size
        padding = [123.675, 116.28, 103.53]
        rh, rw = rgb.shape[:2]
        pad_h = input_size[0] - rh
        pad_w = input_size[1] - rw
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                                pad_w_half, pad_w - pad_w_half,
                                cv.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb_t = torch.div((rgb_t - mean), std)
        rgb_t = rgb_t[None, :, :, :].to(self.device)

        # Inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb_t})

        # Un-pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0] - pad_info[1],
                                pad_info[2]:pred_depth.shape[1] - pad_info[3]]

        # Upsample to original size
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (h, w), mode='bilinear').squeeze()

        # De-canonical transform: convert from canonical (f=1000) to real metric depth
        canonical_to_real = intrinsic_scaled[0] / 1000.0
        pred_depth = pred_depth * canonical_to_real
        pred_depth = torch.clamp(pred_depth, 0, 300)

        log.info(f"Metric3D depth stats: min={pred_depth.min():.2f}, max={pred_depth.max():.2f}, "
                f"mean={pred_depth.mean():.2f}, median={pred_depth.median():.2f}")

        if self.plot:
            plot_depth_maps(self.cfg, img, pred_depth, None)

        return pred_depth

    @staticmethod
    def _patch_metric3d_hub():
        """Patch cached Metric3D hub code to use mmengine instead of mmcv."""
        hub_dir = torch.hub.get_dir()
        comm_path = os.path.join(hub_dir, 'yvanyin_metric3d_main', 'mono', 'utils', 'comm.py')
        if not os.path.exists(comm_path):
            return
        with open(comm_path, 'r') as f:
            lines = f.readlines()

        patched = False
        for i, line in enumerate(lines):
            if line.startswith('from mmcv.utils import collect_env'):
                lines[i:i+1] = [
                    'try:\n',
                    '    from mmcv.utils import collect_env as collect_base_env\n',
                    'except ImportError:\n',
                    '    from mmengine.utils.dl_utils import collect_env as collect_base_env\n',
                ]
                patched = True
                break
        if patched:
            with open(comm_path, 'w') as f:
                f.writelines(lines)
            log.info("Patched Metric3D hub: mmcv → mmengine fallback")

    def _predict_unidepth(self, img):

        if isinstance(img, Image.Image):
            img = np.array(img)

        H, W = img.shape[:2]

        rgb = torch.from_numpy(img).permute(2,0,1).float() / 255.
        rgb = rgb.unsqueeze(0).to(self.device)

        K = torch.tensor(self.K).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model.infer(rgb, K)

        depth = pred["depth"].squeeze()

        if self.plot:
            plot_depth_maps(self.cfg, img, depth, None)

        return depth

@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{cfg}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = load_single_img(cfg)

    de = DepthEstimator(cfg, device, plot=True)
    de._load_model()
    depth = de._predict(img)
    
if __name__ == '__main__':
    main()