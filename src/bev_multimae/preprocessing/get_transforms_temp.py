import numpy as np
from bev_multimae.preprocessing.mcap_reader import list_transforms


################ READ THIS ################
# The way of retrieving transformations work, but the calibration
# was wrong on the current bag file I have been using. Thus, the working code
# has been commented out and in replacement manually written transformation 
# matrices has been placed in their place. For new bag files, the commented code
# should be used. Also, camera frame is the new ego, as I was only provided sensor to 
# camera transformations

def T_cam_to_ego(mcap_path: str, transforms=None) -> np.ndarray:
    if transforms is None:
        transforms = list_transforms(mcap_path, verbose=False)


    chain = [
        ("sensor_base_link",                                 "bracket_front_right"),
        ("bracket_front_right",                              "bracket_camera_front_right"),
        ("bracket_camera_front_right",                       "bracket_camera_front_right_sensor_mounting_point"),
        ("bracket_camera_front_right_sensor_mounting_point", "nominal_camera_front_right"),
        ("nominal_camera_front_right",                       "camera_front_right"),
        ("camera_front_right",                               "camera_front_right_optical_frame"),
    ]

    return chain_transforms(transforms, chain)

def T_rad_to_ego(mcap_path: str, transforms=None) -> np.ndarray:
    if transforms is None:
        transforms = list_transforms(mcap_path, verbose=False)

    chain = [
        ("sensor_base_link",          "bracket_front_right"),
        ("bracket_front_right",       "nominal_radar_front_right"),
        ("nominal_radar_front_right", "radar_front_right"),
    ]
    return chain_transforms(transforms, chain)


def T_lid_to_ego(mcap_path: str, transforms=None) -> np.ndarray:
    if transforms is None:
        transforms = list_transforms(mcap_path, verbose=False)

    chain = [
        ("sensor_base_link",         "nominal_lidar_front_top"),
        ("nominal_lidar_front_top",  "lidar_front_top"),
        ("lidar_front_top",          "lidar_front_top/laser"),
    ]

    return chain_transforms(transforms, chain)

def T_rad_to_cam(mcap_path: str) -> np.ndarray:
    transforms    = list_transforms(mcap_path, verbose=False)
    _T_rad_to_ego = T_rad_to_ego(mcap_path, transforms)
    T_ego_to_cam  = np.linalg.inv(T_cam_to_ego(mcap_path, transforms))

    return T_ego_to_cam @ _T_rad_to_ego

def T_lid_to_cam(mcap_path: str) -> np.ndarray:
    transforms    = list_transforms(mcap_path, verbose=False)
    _T_lid_to_ego = T_lid_to_ego(mcap_path, transforms)
    T_ego_to_cam  = np.linalg.inv(T_cam_to_ego(mcap_path, transforms))

    return T_ego_to_cam @ _T_lid_to_ego

def get_transform(transforms, parent_frame, child_frame):
    key = (parent_frame, child_frame)
    if key not in transforms:
        available = "\n".join(f"  {p} -> {c}" for p, c in transforms.keys())
        raise KeyError(f"Transform {parent_frame} -> {child_frame} not found.\nAvailable:\n{available}")
    return transforms[key]

def chain_transforms(transforms, frame_chain):
    T = np.eye(4)
    for parent, child in frame_chain:
        T = T @ get_transform(transforms, parent, child)
    return T

def apply_transform(T, points_xyz):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N, 1))])
    return (T @ pts_h.T).T[:, :3]
