import os
import torch
import logging
import timm
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)


def main():
    """Extract BEV features from camera BEVs and save to processed data structure."""
    
    # Configuration
    skip = {"meta.pt", "radar_stats.pt"}
    
    target_feat_size = 18  # 18x18 features
    cam_bev_size = 270     # 270x270 camera BEV input
    
    # Load feature extractor once
    log.info("Loading pretrained BEV feature extractor (EfficientNet-B3)...")
    extractor = timm.create_model(
        'efficientnet_b3', 
        pretrained=True, 
        features_only=True, 
        out_indices=(3,)  # stage 3 features (~17x17 from 270x270)
    )
    extractor.eval()
    if torch.cuda.is_available():
        extractor = extractor.cuda()
    
    # Verify output size with dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, cam_bev_size, cam_bev_size)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
        dummy_out = extractor(dummy)[0]
        output_size = dummy_out.shape[-1]
        log.info(f"EfficientNet-B3 stage 3 outputs {output_size}x{output_size} from {cam_bev_size}x{cam_bev_size} input")
    
    print(dummy_out.shape)

    # Process each direction and split
    for direction in ['right', 'left']:
        break
        for split in ['train', 'val']:
            source_dir = f"data/processed_2/{direction}/{split}"
            target_base = f"data/processed_3/{direction}/{split}"
            
            if not os.path.exists(source_dir):
                log.info(f"Skipping {source_dir} (not found)")
                continue
            
            # Find all sample files, sorted by number
            files = sorted(
                [
                    p for p in Path(source_dir).rglob("*.pt")
                    if p.name not in skip
                ],
                key=lambda p: (p.parent.name, int(p.stem.split("_")[-1]))
            )
            
            log.info(f"Processing {len(files)} files from {split} split, {direction} direction")
            
            with torch.no_grad():
                for file_path in tqdm(files, desc=f"{direction}/{split}"):
                    try:
                        # Load sample
                        data = torch.load(file_path, map_location="cpu", weights_only=False)
                        
                        # Extract camera BEV
                        cam_bev = data["cam_bev"].float()  # (3, 270, 270)
                        
                        # Move to GPU and extract features
                        if torch.cuda.is_available():
                            cam_bev = cam_bev.cuda()
                        
                        features = extractor(cam_bev.unsqueeze(0))[0]  # (1, C, H', W')
                        
                        # Resize to target size if needed
                        current_size = features.shape[-1]
                        if current_size != target_feat_size:
                            # features is already (1, C, H, W), pass directly to interpolate
                            features = F.interpolate(
                                features,
                                size=(target_feat_size, target_feat_size),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # Squeeze batch dimension and move to CPU
                        features = features.squeeze(0).cpu()  # (C, H, W)
                        data["bev_feat"] = features
                        
                        # Construct target path preserving event folder structure
                        event_folder = file_path.parent.name
                        filename = file_path.name
                        target_event_dir = os.path.join(target_base, event_folder)
                        os.makedirs(target_event_dir, exist_ok=True)
                        target_path = os.path.join(target_event_dir, filename)
                        
                        # Save to processed_3
                        torch.save(data, target_path)
                        
                    except Exception as e:
                        log.error(f"Error processing {file_path}: {e}")
                        continue
    
    log.info("Feature extraction complete!")


if __name__ == '__main__':
    main()