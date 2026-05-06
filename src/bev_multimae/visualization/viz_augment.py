import matplotlib.pyplot as plt
import torch

def viz_augment(dataset, idx=0):
    data = torch.load(dataset.files[idx], weights_only=False)
    cam_orig = data["cam_bev"].float()
    pts_orig = data["radar"]["points"].clone()

    # force all augmentations on
    orig_h, orig_v, orig_r = dataset.h_flip_rate, dataset.v_flip_rate, dataset.rot_rate
    dataset.h_flip_rate = 1.0
    dataset.v_flip_rate = 1.0
    dataset.rot_rate = 1.0

    cam_aug, radar_aug, target_aug = dataset.augment_sample(cam_orig, data["radar"]["points"])

    dataset.h_flip_rate, dataset.v_flip_rate, dataset.rot_rate = orig_h, orig_v, orig_r

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cam_orig.permute(1, 2, 0).numpy().clip(0, 1), origin='lower')
    axes[0, 0].set_title("Cam BEV original")

    axes[0, 1].imshow(cam_aug.permute(1, 2, 0).numpy().clip(0, 1), origin='lower')
    axes[0, 1].set_title("Cam BEV augmented")

    axes[1, 0].scatter(pts_orig[:, 1].numpy(), pts_orig[:, 2].numpy(), s=2)
    axes[1, 0].set_title("Radar points original")
    axes[1, 0].set_aspect("equal")

    pts_aug = radar_aug["points"]
    axes[1, 1].scatter(pts_aug[:, 1].numpy(), pts_aug[:, 2].numpy(), s=2)
    axes[1, 1].set_title("Radar points augmented")
    axes[1, 1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("augment_check.png")
    print("Saved augment_check.png")