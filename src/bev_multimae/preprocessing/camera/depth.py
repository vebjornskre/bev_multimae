# Get depth code
import torch
import sys
import os
import glob
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F


def cnn_feature_extract(img):

    # Load CNN backbone
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1").to(device)
    model.eval()
    backbone = model.features[:10]

    # Preprocessing
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if isinstance(img, Image.Image):
        transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        x = transform(img).unsqueeze(0).to(device)

    elif isinstance(img, torch.Tensor):
        # Assume tensor is [0, 1] float
        if img.dim() == 3:
            x = normalize(img).unsqueeze(0).to(device)
        elif img.dim() == 4:
            x = normalize(img).to(device)
    
    # Extract feature map
    with torch.no_grad():
        feat = backbone(x)

    print(f"CNN Output shape: {feat.shape}")
    return feat

def zoe_depth(img, feat, plot_save_folder=None, plot=False):

    # Load ZoeDepth-N (metric)
    model = torch.hub.load(
        "isl-org/ZoeDepth",
        "ZoeD_N",
        pretrained=True
    ).to(device)

    model.eval()

    # Inference
    with torch.no_grad():
        depth = model.infer_pil(img, output_type="tensor")  # [1, 1, H, W]

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)

    # We dowsample to the size of feat
    target_size = feat.shape[-2:]

    # Bilinear downsampling to feature maps resolution
    depth_ds = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False, antialias=True)

    if plot:
        # Full image and depth map
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.05})
        
        # Image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Depth
        depth_np = depth.squeeze().cpu().numpy()
        axes[1].imshow(depth_np, cmap='plasma')
        axes[1].set_title("Full Resolution Depth")
        axes[1].axis("off")
        
        plt.savefig(f"{plot_save_folder}/vis_full.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Feature map and downsampled depth
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.05})
        
        feat = feat.squeeze(0)

        # Average all channels or pick a channel
        feat_img = feat.mean(dim=0)
        # feat_img = feat[0,...]

        axes[0].imshow(feat_img, cmap='viridis')
        axes[0].set_title("Feature Input (Tensor)")
        axes[0].axis("off")

        # Downsampled depth
        depth_ds_np = depth_ds.squeeze().cpu().numpy()
        axes[1].imshow(depth_ds_np, cmap='plasma')
        axes[1].set_title("Downsampled Depth")
        axes[1].axis("off")

        plt.savefig(f"{plot_save_folder}/vis_downsampled.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f'Figures saved: {plot_save_folder}/vis_full.png, {plot_save_folder}/vis_downsampled.png')

    return depth_ds

if __name__ == '__main__':
    # Parse CLI arguments
    if len(sys.argv) > 2:
        img_folder = sys.argv[1]
        plot_save_folder = sys.argv[2]
    else:
        raise Exception('You need to provide an image folder and an output folder')
    
    # Check if folder exists
    if not os.path.exists(img_folder):
        print(f"Error: Folder '{img_folder}' does not exist.")
        sys.exit(1)

    # Find images in the folder (jpg, png)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(img_folder, ext)))
    
    if not img_files:
        print(f"Error: No images found in '{img_folder}'.")
        sys.exit(1)

    # Pick the first image found
    img_path = sorted(img_files)[0]
    print(f"Processing image: {img_path}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    img = Image.open(img_path).convert("RGB")

    feat = cnn_feature_extract(img)
    depth_ds = zoe_depth(img, feat, plot_save_folder, plot=True)