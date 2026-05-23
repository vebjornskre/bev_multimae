import torch


@torch.no_grad()
def apply_double_flip_augmentation(detections, point_cloud_range):
    """
    Apply test-time augmentation: average predictions from 4 flipped versions.
    Each flip is done in image space, then predictions are flipped back to original coordinates.

    Args:
        detections: dict with heatmap, reg, height, dim, rot (B, C, H, W)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        dict with averaged detections in original coordinate space
    """
    device = detections['heatmap'].device

    # Original (no flip)
    aug_list = [detections]

    # Horizontal flip: flip W dimension, then flip predictions back
    det_h_flip = {
        'heatmap': torch.flip(detections['heatmap'], dims=[-1]),
        'reg': torch.flip(detections['reg'].clone(), dims=[-1]),
        'height': torch.flip(detections['height'], dims=[-1]),
        'dim': torch.flip(detections['dim'], dims=[-1]),
        'rot': torch.flip(detections['rot'].clone(), dims=[-1]),
    }
    # Flip coordinates back to original space: x offset negated, sin negated
    det_h_flip['reg'][:, 0:1, :, :] *= -1.0  # negate x offset
    det_h_flip['rot'][:, 0:1, :, :] *= -1.0  # negate sin
    # Flip the spatial dimensions back
    det_h_flip['heatmap'] = torch.flip(det_h_flip['heatmap'], dims=[-1])
    det_h_flip['reg'] = torch.flip(det_h_flip['reg'], dims=[-1])
    det_h_flip['height'] = torch.flip(det_h_flip['height'], dims=[-1])
    det_h_flip['dim'] = torch.flip(det_h_flip['dim'], dims=[-1])
    det_h_flip['rot'] = torch.flip(det_h_flip['rot'], dims=[-1])
    aug_list.append(det_h_flip)

    # Vertical flip: flip H dimension, then flip predictions back
    det_v_flip = {
        'heatmap': torch.flip(detections['heatmap'], dims=[-2]),
        'reg': torch.flip(detections['reg'].clone(), dims=[-2]),
        'height': torch.flip(detections['height'], dims=[-2]),
        'dim': torch.flip(detections['dim'], dims=[-2]),
        'rot': torch.flip(detections['rot'].clone(), dims=[-2]),
    }
    # Flip coordinates back to original space: y offset negated, cos negated
    det_v_flip['reg'][:, 1:2, :, :] *= -1.0  # negate y offset
    det_v_flip['rot'][:, 1:2, :, :] *= -1.0  # negate cos
    # Flip the spatial dimensions back
    det_v_flip['heatmap'] = torch.flip(det_v_flip['heatmap'], dims=[-2])
    det_v_flip['reg'] = torch.flip(det_v_flip['reg'], dims=[-2])
    det_v_flip['height'] = torch.flip(det_v_flip['height'], dims=[-2])
    det_v_flip['dim'] = torch.flip(det_v_flip['dim'], dims=[-2])
    det_v_flip['rot'] = torch.flip(det_v_flip['rot'], dims=[-2])
    aug_list.append(det_v_flip)

    # Both flips: flip both dimensions, then flip predictions back
    det_both_flip = {
        'heatmap': torch.flip(detections['heatmap'], dims=[-2, -1]),
        'reg': torch.flip(detections['reg'].clone(), dims=[-2, -1]),
        'height': torch.flip(detections['height'], dims=[-2, -1]),
        'dim': torch.flip(detections['dim'], dims=[-2, -1]),
        'rot': torch.flip(detections['rot'].clone(), dims=[-2, -1]),
    }
    # Flip coordinates back to original space: both offsets negated, both sin/cos negated
    det_both_flip['reg'] *= -1.0
    det_both_flip['rot'] *= -1.0
    # Flip the spatial dimensions back
    det_both_flip['heatmap'] = torch.flip(det_both_flip['heatmap'], dims=[-2, -1])
    det_both_flip['reg'] = torch.flip(det_both_flip['reg'], dims=[-2, -1])
    det_both_flip['height'] = torch.flip(det_both_flip['height'], dims=[-2, -1])
    det_both_flip['dim'] = torch.flip(det_both_flip['dim'], dims=[-2, -1])
    det_both_flip['rot'] = torch.flip(det_both_flip['rot'], dims=[-2, -1])
    aug_list.append(det_both_flip)

    # Average all augmentations in original coordinate space
    augmented_detections = {}
    for key in detections.keys():
        augmented_detections[key] = torch.stack([d[key] for d in aug_list], dim=0).mean(dim=0)

    return augmented_detections


@torch.no_grad()
def circle_nms_torch(boxes, scores, min_radius=1.0, post_max_size=100):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    centers = boxes[:, :2]

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= post_max_size or order.numel() == 1:
            break

        cur = centers[i].view(1, 2)
        rest = centers[order[1:]]
        dist = torch.norm(rest - cur, dim=1)

        order = order[1:][dist > min_radius]

    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def decode_centerpoint(
    detections,
    point_cloud_range,
    score_thresh=0.1,
    post_center_range=None,
    topk=None,
    out_size_factor=None,
    voxel_size=None,
    use_circle_nms=True,
    min_radius=1.0,
    nms_post_max_size=100,
):
    hm = detections["heatmap"]
    hm = hm.sigmoid()
    hm = torch.nn.functional.max_pool2d(hm, kernel_size=3, stride=1, padding=1).eq(hm) * hm

    reg = detections["reg"]
    height = detections["height"]
    dim = detections["dim"]
    rot = detections["rot"]

    B, C, H, W = hm.shape
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    if voxel_size is not None and out_size_factor is not None:
        vx, vy = voxel_size[:2]
        sx = out_size_factor * vx
        sy = out_size_factor * vy
    else:
        sx = (x_max - x_min) / W
        sy = (y_max - y_min) / H

    if post_center_range is None:
        post_center_range = point_cloud_range

    post_center_range = torch.tensor(
        post_center_range,
        dtype=hm.dtype,
        device=hm.device,
    )

    results = []

    for b in range(B):
        hm_b = hm[b].permute(1, 2, 0).contiguous()
        reg_b = reg[b].permute(1, 2, 0).contiguous()
        height_b = height[b].permute(1, 2, 0).contiguous()
        dim_b = dim[b].permute(1, 2, 0).contiguous()
        rot_b = rot[b].permute(1, 2, 0).contiguous()

        scores, labels = torch.max(hm_b.view(-1, C), dim=-1)

        if topk is not None and topk < scores.numel():
            scores, topk_inds = torch.topk(scores, topk)
            labels = labels[topk_inds]
            flat_inds = topk_inds
        else:
            flat_inds = torch.arange(scores.numel(), device=hm.device)

        ys = flat_inds // W
        xs = flat_inds % W

        offsets = reg_b[ys, xs]
        z = height_b[ys, xs]
        size = dim_b[ys, xs]
        r = rot_b[ys, xs]

        x = (xs.float() + offsets[:, 0]) * sx + x_min
        y = (ys.float() + offsets[:, 1]) * sy + y_min
        yaw = torch.atan2(r[:, 0], r[:, 1])

        boxes = torch.cat(
            [
                x[:, None],
                y[:, None],
                z,
                size,
                yaw[:, None],
            ],
            dim=1,
        )

        score_mask = scores > score_thresh
        range_mask = (
            (boxes[:, :3] >= post_center_range[:3]).all(dim=1)
            & (boxes[:, :3] <= post_center_range[3:]).all(dim=1)
        )

        keep = score_mask & range_mask

        # Debug logging
        import logging
        log = logging.getLogger(__name__)
        if len(scores) > 0:
            log.debug(f"Decode: {len(scores)} total boxes, {score_mask.sum()} pass score threshold, {range_mask.sum()} in range, {keep.sum()} kept")
            if keep.sum() == 0 and score_mask.sum() > 0:
                log.warning(f"All {score_mask.sum()} high-confidence boxes filtered by range check!")
                log.warning(f"Post-center-range: {post_center_range}")
                in_range_boxes = boxes[score_mask]
                log.warning(f"First 3 high-score boxes (x,y,z): {in_range_boxes[:3, :3]}")

        boxes = boxes[keep]
        scores_kept = scores[keep]
        labels_kept = labels[keep]

        if boxes.numel() > 0 and use_circle_nms:
            selected = circle_nms_torch(
                boxes,
                scores_kept,
                min_radius=min_radius,
                post_max_size=nms_post_max_size,
            )
            boxes = boxes[selected]
            scores_kept = scores_kept[selected]
            labels_kept = labels_kept[selected]

        results.append(
            {
                "boxes": boxes,
                "scores": scores_kept,
                "labels": labels_kept,
            }
        )

    return results