# Get depth code
import contextlib
import torch
import numpy as np
import sys
import os
import glob
import warnings
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import logging
from pathlib import Path

import cv2 as cv

from transformers import pipeline

log = logging.getLogger(__name__)


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

def zoe_depth(model, img, plot_save_folder=None, plot=False, device='cpu'):

    model.to(device)
    model.eval()

    # Hook into last BEiT block to grab features
    feats = {}
    def hook_fn(module, input, output):
        feats['backbone'] = output

    # Hook to capture the actual input resolution after MiDaS Resize transform
    input_size = {}
    def input_hook_fn(module, inp):
        input_size['processed'] = inp[0].shape  # (B, C, H, W)

    hook = model.core.core.pretrained.model.blocks[23].register_forward_hook(hook_fn)
    input_hook = model.core.core.pretrained.model.patch_embed.register_forward_pre_hook(input_hook_fn)
    depth = model.infer_pil(img, output_type="tensor")
    hook.remove()
    input_hook.remove()

    if 'processed' in input_size:
        log.info(f"ZoeDepth actual processing resolution: {input_size['processed']}")

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)

    # Reshape BEiT tokens to spatial feature map
    feat = feats['backbone']              # (B, N+1, C)
    feat = feat[:, 1:, :]                 # drop CLS token
    B, N, C = feat.shape
    img_w, img_h = img.size              
    ratio = img_w / img_h
    h = int(round((N / ratio) ** 0.5))    # patch grid height
    w = N // h                            # patch grid width
    assert h * w == N, f"Patch grid {h}x{w} != {N} tokens"
    feat = feat.transpose(1, 2).reshape(B, C, h, w)  # (B, C, h, w)

    # Downsample depth to feature map resolution
    target_size = feat.shape[-2:]
    depth_ds = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False, antialias=True)

    log.info(f"ZoeDepth feature shape: {feat.shape}, depth downsampled to: {depth_ds.shape}")

    if plot:
        from bev_multimae.visualization.depth_visualization import plot_depth_maps
        plot_depth_maps(plot_save_folder, img, depth, feat)

    return depth_ds, feat

def load_zoe(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Patch load_state_dict to allow strict=False (timm version mismatch)
    _orig_load = nn.Module.load_state_dict
    nn.Module.load_state_dict = lambda self, state_dict, **kw: _orig_load(self, state_dict, strict=False)

    # Load ZoeDepth (suppress verbose output)
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

def load_single_img(cfg: DictConfig, idx=0, return_path=False):
    img_folder = Path(to_absolute_path(cfg.imgs_raw_path))
    idx = cfg.img_frame

    ext = "*.jpg"
    img_files = glob.glob(os.path.join(img_folder, ext))

    img_path = sorted(img_files)[idx]
    log.info(f"Processing image: {Path(img_path).name}")
    img = Image.open(img_path).convert("RGB")

    if return_path:
        return img, img_path
    return img


def depth_pro_prediction(cfg: DictConfig, image_path:str, plot=False):

    import depth_pro
    from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT
    from dataclasses import replace

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=cfg.depth_pro_weights
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

    # Load and preprocess an image.
    orig_image, _, _ = depth_pro.load_rgb(image_path)

    # Use focal length from calibration instead of EXIF (robot cameras lack valid EXIF data)
    cam_info = np.load(cfg.camera_info)
    f_px = cam_info['K'][0, 0]
    log.info(f"Using calibrated focal length: f_px={f_px:.2f}")

    image = transform(orig_image)
    image = image.to(device)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    plot_save_folder = cfg.plot_folder

    if plot:
        from bev_multimae.visualization.depth_visualization import plot_depth_maps
        plot_depth_maps(plot_save_folder, orig_image, depth, None)
    
    return depth

def depth_anything_prediction(cfg: DictConfig, img, plot=False):
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run inference on a PIL image
    result = pipe(img)  # img is a PIL.Image
    depth = result["depth"]  # PIL Image of depth
    depth_tensor = result["predicted_depth"]  # torch.Tensor [1, H, W] in meters

    plot_save_folder = cfg.plot_folder

    if plot:
        from bev_multimae.visualization.depth_visualization import plot_depth_maps
        plot_depth_maps(plot_save_folder, img, depth_tensor, None)

    return depth_tensor
    

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


def metric3d_prediction(cfg: DictConfig, img, plot=False):
    """Metric3D v2 depth prediction with canonical camera space transform."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Patch cached hub code to avoid mmcv dependency
    _patch_metric3d_hub()

    # Load model 
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
    model.to(device).eval()
    log.info("Metric3D v2 ViT-Large model loaded")

    ## Data transformations
    cam_info = np.load(cfg.camera_info)
    K = cam_info['K']
    intrinsic = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # [fx, fy, cx, cy]

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
    rgb_t = rgb_t[None, :, :, :].to(device)

    # Inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb_t})

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

    if plot:
        from bev_multimae.visualization.depth_visualization import plot_depth_maps
        plot_depth_maps(cfg.plot_folder, img, pred_depth, None)

    return pred_depth


@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    '''
    Functions from this file should be imported to other files, 
    if this file is run directly it will only process and plot 
    one image, then save it.
    '''
    
    log.info(f"Configuration:\n{cfg}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img, path = load_single_img(cfg, return_path=True)

    if cfg.depth_model == 'zoe':
        zoe = load_zoe(cfg)
        plot_folder = os.path.join(Path(to_absolute_path(cfg.plot_folder)), 'depth_imgs')
        depth_ds, feat = zoe_depth(zoe, img=img, plot_save_folder=plot_folder, plot=True, device=device)

    elif cfg.depth_model == 'depth_pro':
        depth = depth_pro_prediction(cfg, path, plot=True)

    elif cfg.depth_model == 'depth_any':
        depth = depth_anything_prediction(cfg, img, plot=True)

    elif cfg.depth_model == 'metric3d':
        depth = metric3d_prediction(cfg, img, plot=True)

    else: raise RuntimeError('Enter valid model name in config')

if __name__ == '__main__':
    main()